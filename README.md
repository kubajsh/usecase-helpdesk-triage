 # GRC Incident Assigner

## About the Project
This project is a simple web-based tool to automatically assign GRC (Governance, Risk, and Compliance) incidents to the correct resolver group. It uses a machine learning model (TF-IDF and Cosine Similarity) to find the most similar past incidents from a knowledge base
and predicts the assignment group based on that.

## Key Features

**Automated Incident Assignment:** Predicts the correct resolver group for GRC incidents based on their description.
**Justification:** Shows the top 3 most similar past incidents to justify the prediction.
**Visual Workflow:** Displays a sample lifecycle of a similar past ticket.
**Simple Web Interface:** Easy-to-use interface to enter an incident description and see the results.
**Flask Backend:** A lightweight Python backend to handle the logic.
**No Database Required:** Uses a hardcoded knowledge base for simplicity.

   How It Works

1.  **Knowledge Base:** A list of past GRC incidents with their descriptions and resolver groups is hardcoded into the Python script.
2.  **TF-IDF Vectorization:** The descriptions in the knowledge base are converted into a matrix of TF-IDF features.
3.  **User Input:** The user enters a new incident description in the web interface.
4.  **Cosine Similarity:** The new description is vectorized, and its cosine similarity is calculated against all the incidents in the knowledge base.
5.  **Prediction:** The incident with the highest similarity score is used to predict the resolver group for the new incident.
6.  **Display Results:** The predicted group, along with the top 3 similar incidents and a sample workflow, are displayed to the user.

## Getting Started
  
### Running with Docker

This is the recommended way to run the application.

1.  Clone the repository:
      git clone https://github.com/kubajsh/usecase-helpdesk-triage.git
2.  Navigate to the project directory:
      cd usecase-helpdesk-triage
3.  Build the Docker image:
      docker build -t grc-incident-assigner .

4.  Run the Docker container:
      docker run -p 8080:8080 grc-incident-assigner

5.  Open your web browser and go to `http://127.0.0.1:8080`

## How to Use

1.  Enter a description of the GRC incident in the text area.
2.  Click the "Submit" button.
3.  The predicted assignment group will be displayed.
4.  Below the prediction, you will see the top 3 most similar incidents from the knowledge base that were used to make the prediction.
5.  On the right, you will see a sample lifecycle of a similar past ticket.


**Customer Workflow: Using the GRC Incident Assigner**

This section explains how a user, such as a help desk agent or IT support staff, would use the GRC Incident Assigner tool to triage incoming incidents efficiently.

The Goal

The primary goal of this tool is to reduce the time and guesswork involved in assigning a new Governance, Risk, and Compliance (GRC) incident to the correct specialized team. By providing an instant, data-driven recommendation, it ensures incidents get to the right experts faster.

Step-by-Step Usage

Imagine a help desk agent has just received a new ticket or an email from an employee with the following description:

"My project team needs to use a new piece of software that isn't on the approved list. We need to get an exception to the standard software policy."

  Here is how the agent would use the tool:

  Open the Tool: The agent navigates to the GRC Incident Assigner web page in their browser.

  Enter the Description: The agent copies the description from the ticket and pastes it into the large text area on the page.

  Submit for Analysis: The agent clicks the "Submit" button.

Receive Instant Recommendation: In a few moments, the tool provides a clear recommendation.
       * Predicted Group: The most prominent result is the name of the team the ticket should be assigned to. For the example above, it would likely predict: CUSTOMER_GRC_ANALYSTS_24.

Understand the Justification: To build confidence and provide transparency, the tool also shows why it made that recommendation.
       *** Similar Tickets:** The agent sees a list of the top 3 most similar incidents from the past. They might see a past ticket with the description: "A user requests to be exempt from a security policy (e.g., 'Need to use a non-standard app')."
       *** Similarity Score:** Each similar ticket has a score, showing how closely it matches the new incident. This helps the agent quickly validate the tool's logic.

Assign the Ticket: Confident in the recommendation, the agent assigns the new incident to the CUSTOMER_GRC_ANALYSTS_24 group in their IT Service Management (ITSM) system.

Key Benefits for the Customer

* Speed: Reduces triage time from minutes to seconds.
* Accuracy: Minimizes human error and mis-routed tickets, which prevents delays in incident resolution.
* Consistency: Ensures that similar types of incidents are always routed to the same team, leading to a more standardized process.
* Empowerment: Enables even junior help desk staff to make accurate assignment decisions without needing to escalate or consult senior colleagues.
* Transparency: The justification section helps users trust the tool and understand the reasoning behind its suggestions.

