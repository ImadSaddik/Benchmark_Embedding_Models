import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE


def load_data_by_language(
    embeddings_directory: str, language: str = "english"
) -> list[dict]:
    questions = []
    chunk_id_counter = 0
    file_paths = _get_file_paths(embeddings_directory, language, sort=True)

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        old_to_new_id_map = {}
        file_chunks = data["chunks"]
        for chunk in file_chunks:
            old_id = chunk["id"]
            new_id = chunk_id_counter
            old_to_new_id_map[old_id] = new_id
            chunk_id_counter += 1

        file_questions = data["question_answer_pairs"]
        for question in file_questions:
            old_chunk_id = question["chunk_id"]
            if old_chunk_id in old_to_new_id_map:
                question["chunk_id"] = old_to_new_id_map[old_chunk_id]
                questions.append(question)

    print(f"{language.capitalize()} - Questions: {len(questions)}")
    return questions


def _get_file_paths(embeddings_directory: str, language: str, sort: bool = False):
    file_paths = []

    for file in os.listdir(embeddings_directory):
        if not file.endswith(".json"):
            continue

        if language == "english":
            languages = ["arabic"]
            if not any(language in file.lower() for language in languages):
                file_paths.append(os.path.join(embeddings_directory, file))
        else:
            if language.lower() in file.lower():
                file_paths.append(os.path.join(embeddings_directory, file))

    if sort:
        file_paths.sort()

    return file_paths


def get_embeddings_from_questions(questions: list[dict], model_name: str) -> np.ndarray:
    all_embeddings = []
    for question in questions:
        all_embeddings.append(question["embeddings"][model_name])

    all_embeddings = np.array(all_embeddings)
    return all_embeddings


def get_sample_indices(sample_size: int, questions_english: list[dict]) -> list[int]:
    english_indices = list(range(sample_size))
    arabic_indices = list(
        range(len(questions_english), len(questions_english) + sample_size)
    )
    sample_indices = english_indices + arabic_indices
    return sample_indices


def plot_data(
    sample_size: int,
    embeddings_2d: np.ndarray,
    questions: list[dict],
    save_figure: bool = False,
    save_path: str = "english_arabic_concept_same_location_vector_space.svg",
) -> None:
    labels = ["English"] * sample_size + ["Arabic"] * sample_size
    df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "label": labels,
            "question": [question["question"] for question in questions],
        }
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        title=f"2D t-SNE visualization of {sample_size} English and {sample_size} Arabic questions",
        labels={"x": "t-SNE component 1", "y": "t-SNE component 2"},
        hover_data=["question"],
    )

    if save_figure:
        fig.write_image(save_path)

    fig.show()


def main() -> None:
    embeddings_directory = "../data/embeddings/"
    questions_arabic = load_data_by_language(embeddings_directory, "arabic")
    questions_english = load_data_by_language(embeddings_directory, "english")

    questions = questions_english + questions_arabic
    embeddings = get_embeddings_from_questions(questions, "qwen3-embedding-4b")

    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings_2d = tsne.fit_transform(embeddings)

    sample_size = 10
    sample_indices = get_sample_indices(sample_size, questions_english)
    sample_embeddings_2d = all_embeddings_2d[sample_indices]
    sample_questions = [questions[i] for i in sample_indices]

    plot_data(
        sample_size=sample_size,
        embeddings_2d=sample_embeddings_2d,
        questions=sample_questions,
        save_figure=True,
    )


if __name__ == "__main__":
    main()
