import os
import re
import json
import argparse
import torch
from typing import List, Tuple, Dict, Any, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- PROMPTS ---

PROMPT_EN = """
Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.

Process Overview: You will be presented with:

1. The criteria for making a judgment.
2. The reference radiology report.
3. The candidate radiology report.
4. The desired format for your assessment.

CRITICAL RULE ABOUT NEGATIVES / NORMAL FINDINGS:
    - Do NOT penalize the candidate for omitting explicit negative/normal statements.
        Example: if the reference says "No pneumothorax" but the candidate does not mention pneumothorax at all, that is NOT an error.
    - Only count an error about a negative/normal finding if the candidate explicitly CONTRADICTS it
        (e.g., reference: "No pneumothorax" but candidate: "Pneumothorax present").

1. Criteria for Judgment:

    For each candidate report, determine:

    The count of errors.

    Errors can fall into one of these categories:

    a) False report of a finding in the candidate.
    b) Missing a finding present in the reference.
    c) Misidentification of a finding's anatomic location/position.
    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.

2. Reference Report:
    {text_ref}

3. Candidate Report:
    {text_cand}

    4. Reporting Your Assessment:

    Follow this specific format for your output, even if no errors are found:
    ```
    [Explanation]:
    <Explanation>

    [Errors]:
    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>
    ....
    (c) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>

    [Matched Findings]:
    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>
    ```
"""

PROMPT_RU = """ 
Задача: Оценить точность предлагаемого радиологического отчета по сравнению с эталонным отчетом, составленным экспертами-рентгенологами.

Процесс: Тебе будут представлены:
1. Критерии для оценки.
2. Эталонный радиологический отчет.
3. Оцениваемый радиологический отчет.
4. Требуемый формат для вашего заключения.

ВАЖНЕЙШЕЕ ПРАВИЛО ОБ ОТРИЦАТЕЛЬНЫХ / НОРМАЛЬНЫХ НАХОДКАХ:
НЕ считай ошибкой, если в оцениваемом отчете отсутствует явное упоминание об отсутствии патологии (отрицательное утверждение).
Пример: если в эталоне указано "Пневмоторакса нет", а в оцениваемом отчете пневмоторакс не упомянут вообще — это НЕ ошибка.
Ошибкой считается только явное противоречие отрицательному утверждению.
Пример: в эталоне "Пневмоторакса нет", а в оцениваемом отчете "Присутствует пневмоторакс".

1. Критерии оценки
Для каждого оцениваемого отчета определи количество ошибок.

Ошибки делятся на категории:

a) Ложное описание: Наличие в оцениваем отчете находки (патологии), которой нет в эталоне.
b) Пропуск находки: Отсутствие в оцениваем отчете находки (патологии), которая есть в эталоне.
c) Ошибка локализации: Неверное указание анатомического расположения находки.

Примечание: Сосредоточься на клинически значимых находках, а не на стиле изложения. Оценивай только те находки, которые так или иначе фигурируют в обоих отчетах (явно или по правилу о противоречии для отрицательных утверждений).

2. Эталонный отчет:
{text_ref}

3. Оцениваемый отчет:
{text_cand}

4. Формат вывода:

Строго следуй указанному формату вывода, даже если ошибок не найдено:

```
[Объяснение]:
<Объяснение>

[Ошибки по категориям]:
(a) Ложное описание: <количество>. <Ошибка 1>; <Ошибка 2>; ...; <Ошибка N>
(b) Пропуск находки: <количество>. <Ошибка 1>; <Ошибка 2>; ...; <Ошибка N>
(c) Ошибка локализации: <количество>. <Ошибка 1>; <Ошибка 2>; ...; <Ошибка N>

[Совпадающие находки]:
<количество совпадений>. <Находка 1>; <Находка 2>; ...; <Находка N>

```
"""

_A_RE = re.compile(r"^\s*\(a\).*?\:\s*(\d+)\b", re.IGNORECASE | re.MULTILINE)
_B_RE = re.compile(r"^\s*\(b\).*?\:\s*(\d+)\b", re.IGNORECASE | re.MULTILINE)
_C_RE = re.compile(r"^\s*\(c\).*?\:\s*(\d+)\b", re.IGNORECASE | re.MULTILINE)

_MATCH_EN_RE = re.compile(r"\[Matched Findings\]:?\s*\n\s*(\d+)[.]?", re.IGNORECASE | re.MULTILINE)
_MATCH_RU_RE = re.compile(r"\[Совпадающие находки\]:?\s*\n\s*(\d+)[.]?", re.IGNORECASE | re.MULTILINE)


def parse_model_output(output_text: str, language: str = 'ru') -> Dict[str, int]:
    """
    Parses model output to find the amount of errors and matched findings.
    """
    def _get_int(rx: re.Pattern, s: str) -> int:
        m = rx.search(s or "")
        if not m:
            return 0
        try:
            return int(m.group(1))
        except Exception:
            return 0

    a = _get_int(_A_RE, output_text)
    b = _get_int(_B_RE, output_text)
    c = _get_int(_C_RE, output_text)
    
    if language == 'ru':
        matched = _get_int(_MATCH_RU_RE, output_text)
    else:
        matched = _get_int(_MATCH_EN_RE, output_text)
        if matched == 0:
             matched = _get_int(_MATCH_RU_RE, output_text)

    return {
        'a': a,
        'b': b,
        'c': c,
        'matched_findings': matched,
        'total_errors': a + b + c
    }


def calculate_report_score(parsed_output: Dict[str, Any]) -> float:
    """
    Calculates score with formula:
    matched_findings / (total_errors + matched_findings)
    """
    total_errors = parsed_output['total_errors']
    matched_findings = parsed_output['matched_findings']

    if matched_findings == 0:
        return 0.0
    
    denominator = total_errors + matched_findings
    score = matched_findings / denominator
    
    return score


def estimate_report_quality(
    candidate_reports: List[str],
    ground_truth_reports: List[str],
    path_to_llm: str = 'Qwen/Qwen2.5-72B-Instruct-AWQ',
    language: str = 'ru'
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates reports using LLM.

    Args:
        candidate_reports: List of candidate reports.
        ground_truth_reports: List of reference reports.
        path_to_llm: Path to the model or ID on HuggingFace.
        language: 'ru' or 'en'.

    Returns:
        Average score for all reports,
        List of dicts with results for each report pair.
    """
    
    if len(candidate_reports) != len(ground_truth_reports):
        raise ValueError("The amount of candidate and reference reports must be the same.")

    prompt_template = PROMPT_RU if language == 'ru' else PROMPT_EN

    print(f"Loading model: {path_to_llm}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path_to_llm, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path_to_llm,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    elementwise_results = []
    total_score = 0.0

    print("Starting evaluation...")
    for cand, ref in tqdm(zip(candidate_reports, ground_truth_reports), total=len(candidate_reports)):
        formatted_prompt = prompt_template.format(text_ref=ref, text_cand=cand)
        messages = [
            {"role": "system", "content": "You are an expert chest radiologist."},
            {"role": "user", "content": formatted_prompt},
        ]

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0, 
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed = parse_model_output(output_text, language=language)
        score = calculate_report_score(parsed)

        result_item = {
            'score': score,
            'metrics': parsed,
            'report': output_text,
            'candidate': cand,
            'ground_truth': ref
        }
        elementwise_results.append(result_item)
        total_score += score

    average_score = total_score / len(candidate_reports) if candidate_reports else 0.0
    
    return average_score, elementwise_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Radiology Reports using an LLM.")
    
    parser.add_argument(
        "--candidates", 
        type=str, 
        required=True, 
        help="Path to JSON file containing a list of candidate reports strings."
    )
    parser.add_argument(
        "--ground_truth", 
        type=str, 
        required=True, 
        help="Path to JSON file containing a list of ground truth reports strings."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="evaluation_results.json", 
        help="Path to save the output JSON."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen2.5-72B-Instruct-AWQ", 
        help="HuggingFace model ID or local path."
    )
    parser.add_argument(
        "--language", 
        type=str, 
        choices=['ru', 'en'], 
        default='ru', 
        help="Language of the reports and prompts (ru/en)."
    )

    args = parser.parse_args()

    try:
        with open(args.candidates, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        with open(args.ground_truth, 'r', encoding='utf-8') as f:
            ground_truths = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        return

    if not isinstance(candidates, list) or not isinstance(ground_truths, list):
        print("Error: Input JSON files must contain a list of strings.")
        return

    avg_score, details = estimate_report_quality(
        candidate_reports=candidates,
        ground_truth_reports=ground_truths,
        path_to_llm=args.model_path,
        language=args.language
    )

    output_data = {
        'average_score': avg_score,
        'elementwise_score': [
            {'score': item['score'], 'report': item['report'], 'metrics': item['metrics']} 
            for item in details
        ]
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    print(f"Evaluation complete. Average Score: {avg_score:.4f}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()