# RAG (Retrieval Augmented Generation)-learning DistilGPT-2 on the 'Diseases_Symptoms' Dataset

This project focuses on RAG-learning of a `distilgpt2` model to generate meaningful text based on disease names, symptoms, and treatments. The dataset used for training includes structured information about diseases, their symptoms, and corresponding treatments.

## Features
- **Dataset:** Utilizes the `QuyenAnhDE/Diseases_Symptoms` dataset from Hugging Face's `datasets` library.
- **Model:** Fine-tunes the `distilgpt2` model from Hugging Face's `transformers` library.
- **Training:** Implements customized data preprocessing and training pipelines using PyTorch.
- **Text Generation:** Generates coherent text given a disease name as input.

## Usage
1. **Install Dependencies:**
   Make sure you have the necessary Python libraries installed:
   ```bash
    pip install transformers datasets torch tqdm sentencepiece pandas
   ```

2. **Train the model:**
    Follow the `rag_llm.ipynb` step by step in order to train the model.

3. **Generate text:**
    After training, provide a disease name as input to generate corresponding symptoms and treatments:
    ```python
        input_str = "Panic disorder"
        input_ids = tokenizer.encode(input_str, return_tensors='pt').to(device)
        output = model.generate(input_ids, max_length=70, do_sample=True, top_k=10, top_p=0.8)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(decoded_output)
    ```

## Results
- The model successfully learns to generate relevant information about diseases, symptoms, and treatments in a structured format.
- Training and validation loss decrease steadily, indicating proper convergence.

## Visualization
- Training and validation loss metrics are visualized using line plots for better interpretability.
    ![Example](/loss.png)
- The generated text is coherent and follows the expected format.

**Input:**

Cellulitis

**Output:**
```basic
    Cellulitis | Redness, Pain, Swelling, Skin changes, Lymph node enlargement | Antibiotics (usually penicillin), Warm compresses, rest, immobilization
```

## Author

Developed by Saveliy Maksimau. If you have any questions or feedback, feel free to reach out or create an issue in the repository.