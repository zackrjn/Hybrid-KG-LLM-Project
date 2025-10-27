from typing import List, Tuple, Optional


def build_text_prompt(query: str,
                      selected_neighbors: List[Tuple[str, str, str]],
                      instruction: Optional[str] = None) -> str:
    lines: List[str] = []
    if instruction:
        lines.append(instruction)
    lines.append("Question: " + query)
    if selected_neighbors:
        lines.append("Relevant KG triples:")
        for h, r, t in selected_neighbors:
            lines.append(f"- ({h}) -[{r}]-> ({t})")
    lines.append("Answer:")
    return "\n".join(lines)


def build_multimodal_prompt(query: str,
                            image_path: Optional[str],
                            selected_neighbors: List[Tuple[str, str, str]],
                            instruction: Optional[str] = None) -> dict:
    text = build_text_prompt(query, selected_neighbors, instruction)
    prompt: dict = {"text": text}
    if image_path:
        prompt["image"] = image_path
    return prompt


