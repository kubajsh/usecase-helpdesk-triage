# GRC Incident Assigner

## About the Project

This project is a simple web-based tool to automatically assign GRC (Governance, Risk, and Compliance) incidents to the correct resolver group. It uses a machine learning model (TF-IDF and Cosine Similarity) to find the most similar past incidents from a knowledge base and predicts the assignment group based on that.

## Key Features

-   **Automated Incident Assignment:** Predicts the correct resolver group for GRC incidents based on their description.
-   **Justification:** Shows the top 3 most similar past incidents to justify the prediction.
-   **Visual Workflow:** Displays a sample lifecycle of a similar past ticket.
-   **Simple Web Interface:** Easy-to-use interface to enter an incident description and see the results.
-   **Flask Backend:** A lightweight Python backend to handle the logic.
-   **No Database Required:** Uses a hardcoded knowledge base for simplicity.

## How It Works

1.  **Knowledge Base:** A list of past GRC incidents with their descriptions and resolver groups is hardcoded into the Python script.
2.  **TF-IDF Vectorization:** The descriptions in the knowledge base are converted into a matrix of TF-IDF features.
3.  **User Input:** The user enters a new incident description in the web interface.
4.  **Cosine Similarity:** The new description is vectorized, and its cosine similarity is calculated against all the incidents in the knowledge base.
5.  **Prediction:** The incident with the highest similarity score is used to predict the resolver group for the new incident.
6.  **Display Results:** The predicted group, along with the top 3 similar incidents and a sample workflow, are displayed to the user.

## Getting Started

### Prerequisites

-   Python 3
-   pip

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    ```
2.  Navigate to the project directory:
    ```sh
    cd your-repository-name
    ```
3.  Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```
4.  Run the Flask application:
    ```sh
    python app.py
    ```
5.  Open your web browser and go to `http://127.0.0.1:8080`

## How to Use

1.  Enter a description of the GRC incident in the text area.
2.  Click the "Submit" button.
3.  The predicted assignment group will be displayed.
4.  Below the prediction, you will see the top 3 most similar incidents from the knowledge base that were used to make the prediction.
5.  On the right, you will see a sample lifecycle of a similar past ticket.

## Future Improvements

-   **Use a Database:** Replace the hardcoded knowledge base with a database (e.g., PostgreSQL, MySQL, or a NoSQL database) to store and manage incidents.
-   **More Sophisticated Model:** Implement a more advanced NLP model (e.g., using word embeddings like Word2Vec or a transformer-based model like BERT) for better prediction accuracy.
-   **User Interface:** Enhance the user interface with more features, such as a dashboard to view past predictions, analytics, and a way to provide feedback on the predictions.
-   **Authentication:** Add user authentication and authorization to control access to the tool.
-   **CI/CD:** Implement a CI/CD pipeline to automate testing and deployment.
-   **Containerization:** Use Docker to containerize the application for easier deployment and scalability.
