# 📧 Spam Email Classification API (MLOps Pipeline)

This repository contains an end-to-end Machine Learning pipeline for **Spam Email Classification**, taking the project from model training to a deployed API with Continuous Integration (CI). 

This project was built to fulfill the assessment requirements: training a model, saving it as a `.joblib` file, serving it via FastAPI, containerizing it with Docker, and automating the build/test process using GitHub Actions.

---

## 📌 Assessment Checklist Mapping

Here is how this repository fulfills the 8 required steps:

1. **Pick a Machine Learning Case Study:** Used a Kaggle dataset (`email_spam.csv`) for Email Spam Classification.
2. **Train and test model on Jupyter/Colab:** The model was trained in Google Colab using a `TfidfVectorizer` and `LogisticRegression` pipeline. 
3. **Save the model:** The entire pipeline (vectorizer + model) was saved as `model.joblib` to ensure consistent text preprocessing during inference.
4. **Test the model with FastAPI (in VS Code):** Built an API in `app.py` that accepts raw email text via a POST request and returns `spam` or `ham`.
5. **Execute API through Docker:** Created a `Dockerfile` to containerize the FastAPI application, ensuring it runs identically across all environments.
6. **Create a GitHub repo:** This repository contains all source code, the Dockerfile, the trained model, and the CI configuration.
7. **Create GitHub Actions CI Pipeline:** Added a `.github/workflows/ci.yml` file to automate the build and testing process.
8. **Make sure the pipeline executed successfully:** The CI pipeline automatically builds the Docker image, runs the container, and tests the `/health` and `/predict` endpoints on every push to `main`.
