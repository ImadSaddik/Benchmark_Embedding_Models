import numpy as np
import pandas as pd
import plotly.express as px
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm


def embed_text(
    model: SentenceTransformer, text: str, prompt_name: str
) -> list[float] | None:
    embedding = None
    embedding = model.encode(sentences=text, prompt_name=prompt_name).tolist()
    if not embedding:
        raise ValueError("No embedding returned from the model.")

    return embedding


def load_model(model_name: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    model = model.to(device)
    return model


def lower_embedding_dimension(all_embeddings: np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, random_state=42)
    all_embeddings_2d = tsne.fit_transform(all_embeddings)
    return all_embeddings_2d


def plot_embeddings(
    all_embeddings_2d: np.ndarray,
    labels: list[str],
    all_sentences: list[str],
    save_plot: bool = False,
    save_path: str = "embedding_clusters.svg",
) -> None:
    df = pd.DataFrame(
        {
            "x": all_embeddings_2d[:, 0],
            "y": all_embeddings_2d[:, 1],
            "label": labels,
            "sentence": all_sentences,
        }
    )
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        title="2D visualization of query embeddings using t-SNE",
        labels={"x": "t-SNE component 1", "y": "t-SNE component 2"},
        color_discrete_sequence=px.colors.qualitative.Plotly,
        opacity=0.7,
        width=800,
        height=600,
        hover_data=["sentence"],
    )

    fig.update_layout(legend_title_text="Query domain")
    if save_plot:
        fig.write_image(save_path)

    fig.show()


space_related_sentences = [
    "What is a black hole?",
    "How many planets are in our solar system?",
    "What is the speed of light?",
    "Tell me about the Apollo 11 mission.",
    "What is the James Webb Space Telescope?",
    "Explain the theory of general relativity.",
    "What are neutron stars?",
    "How was the Moon formed?",
    "What is the Kuiper Belt?",
    "Who was the first person in space?",
    "Describe the surface of Mars.",
    "What are gravitational waves?",
    "What is dark matter?",
    "How do stars form?",
    "What is the Big Bang theory?",
    "Tell me about the Voyager probes.",
    "What is a supernova?",
    "How far is the Andromeda galaxy?",
    "What is the composition of Jupiter's atmosphere?",
    "Explain the concept of a wormhole.",
]
cooking_related_sentences = [
    "How to bake chocolate chip cookies?",
    "What is the best recipe for lasagna?",
    "How to make sourdough bread from scratch?",
    "What are the ingredients for a classic margarita?",
    "How to cook a perfect steak?",
    "What is the recipe for French onion soup?",
    "How to prepare sushi at home?",
    "What is the difference between baking soda and baking powder?",
    "How to make homemade pasta?",
    "What is a good marinade for chicken?",
    "How to make a classic Caesar salad dressing?",
    "What are the steps to brew beer at home?",
    "How to make pickles?",
    "What's the recipe for a pumpkin spice latte?",
    "How to properly roast vegetables?",
    "What is the mother sauce in French cuisine?",
    "How to make gnocchi?",
    "What is the best way to cook salmon?",
    "How to make a vegan chili?",
    "What are the ingredients for a traditional paella?",
]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("Qwen/Qwen3-Embedding-0.6B", device)
    print(f"Model running on {device}.")

    prompt_name = "query"
    all_embeddings = []
    all_sentences = space_related_sentences + cooking_related_sentences

    for sentence in tqdm(all_sentences, total=len(all_sentences)):
        text_chunk_embedding = embed_text(model, sentence, prompt_name)
        all_embeddings.append(text_chunk_embedding)
    print("All sentences have been embedded.")

    all_embeddings = np.array(all_embeddings)
    all_embeddings_2d = lower_embedding_dimension(all_embeddings)
    print("Dimensionality reduction completed.")

    labels = ["Space"] * len(space_related_sentences)
    labels += ["Cooking"] * len(cooking_related_sentences)
    plot_embeddings(
        all_embeddings_2d,
        labels,
        all_sentences,
        save_plot=True,
        save_path="embedding_clusters.svg",
    )
