# Pdf-Chatbot
A complete Local RAG app which helps you converse with pdf files.

<img src = "App UI.png"></img>

## Introduction
This is a local RAG application which helps you converse with a pdf file. Idea is to have a bot which can answer questions using information from pdf file as context and provide responses to user in english language. 

One of the use case could be work with financial, medical or confidential reports which are not supposed to be uploaded anywhere.

## Installation
Since the application runs in local, It expects these dependencies installed prior:

- Download and install Ollama from https://ollama.com/
- 4. We are using Nomic-Embed-text model from Ollama, follow the below instruction to download model into local
    ```    
    ollama pull nomic-embed-text
    ```
    5. Similarly, Download the Gemma 2b model from Ollama.
    ```
    ollama pull gemma:2b
    ```


## How it works
1. Clone the repository.
2. Copy and paste all the documents that you want to work with in the Documents folder, the app will only work with documents which are present in the Documents folder.
3. Install all the dependencies from 'requirements.txt' file by running the following command:
```
pip install -r requirements.txt
```
4. Run the below command from the same directory where the Home.py file is located:
```
streamlit run /Home.py
```
5. The application will launch in your default web browser.
6. Select the document which you want to converse with and click on the process button.
7. The app will check if the embeddings for the file exists or not, if not it will create one for the document for later use.
8. The app will inform you once the embeddings are ready to use.
9. Now, you can converse with the document using natural language by asking questions.

## Credits
The initial codebase is inspired from the following tutorials, I highly recommend you to check out awesome developers behind these tutorials and resources:

1. <a href = "https://www.youtube.com/@alejandro_ao">Alejandro AO - Software & Ai</a> : <a href = "https://youtu.be/dXxQ0LR-3Hg">Tutorial Link</a>
2. <a href = "https://www.youtube.com/@tonykipkemboi">The How-To Guy</a> : <a href = "https://youtu.be/ztBJqzBU5kc">Tutorial Link</a>
3. <a href = "https://ollama.com/">Ollama</a> : For good repo of models which can be downloaded locally.
4. <a href = "https://www.langchain.com/">Langchain</a>: For their AI framework for RAG based applications.

## Contribution
The codebase is free for use and play around with and open to any contribution.