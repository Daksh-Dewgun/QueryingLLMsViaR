# Querying LLMs Via R: A Guide for Researchers

# Introduction

This handbook provides a step-by-step guide for researchers aiming to leverage the capabilities of Large Language Models (LLMs) via R programming. LLMs are an increasingly powerful tool for researchers in a variety of fields with the ability to assist with text-generation, data analysis, classifications, synthesising documents and much more. LLMs have made significant strides in Natural Language Processing (NLP) and are now accessible to researchers through various APIs. This document discusses the ways in which researchers can utilise LLMs, focussing mainly on OpenAI's API - ChatGPT, via different R packages. These packages include `httr` with `jsonlite`, `openai` and `chatgpt`. While most LLMs typically require interfacing with an API, it is also possible to query local LLMs using similar ways, albeit with a little more setup and computing resources at the researcher's dispense. Starting with the pre-requisites for querying LLMs, this handbook provides step-by-step instructions to utilise different R packages for research. Finally, the document discusses some considerations for researchers to keep in mind while conducting this type of research.

# Pre-requisites

**1. R Installation:** Ensure you have R installed on your system. It can be downloaded from [CRAN](https://cran.r-project.org/).

**2. RStudio:** RStudio acts as an Integrated Development Environment (IDE) for R. While it is optional and other softwares may be employed, RStudio is highly recommended for its ease of use.

**3. API Key:** In order to interface most LLMs, you require access to an LLM API Key. They can be obtained directly from the official organisation by signing up and requesting an API key. Some common examples are provided below.

  - OpenAI: (https://platform.openai.com/api-keys)
  - Hugging Face: (https://huggingface.co/)
  - Cohere LLM: (https://cohere.com/)

**Note.** API Keys are unique and must not be shared with others. They must be secured safely and never deployed on client-side environments. If the researcher is using LLMs to debug code, be sure of hiding the API Key before sharing it online. This is because it may be against the Terms of Use to share API Keys. For best practices to keep your API Key secure, read OpenAI's page [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

**4. Internet Access:** Querying LLMs requires making HTTP requests to external APIs. Hence, a stable internet connection is necessary for this purpose.

# 1. Using `httr` with `jsonlite`

The most common way of querying ChatGPT or other LLMs is by way of utilising the `httr` and `jsonlite` packages. While `httr` provides a variety of options to handle HTTP requests, `jsonlite` allows researchers to parse the JSON responses (most common LLM response format) into usable R objects. In addition, `jsonlite` also helps create the JSON request body through the `toJSON()`function. The key benefit of this combination of packages is the flexibility and stability it provides to use different cloud-based or remote LLMs, without restricting the researcher to OpenAI's ChatGPT. 

## Setting up the Environment

**1. Install Packages:** The `httr` and `jsonlite` packages can be installed using the following code:

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE}
# Install Packages
install.packages("httr")
install.packages("jsonlite")
```

**2. Set API Key:** Once these packages are installed, remember to load them and set your API Key.

```{r, echo=TRUE, results='hide', eval=FALSE}
# Load Packages
library(httr)
library(jsonlite)

# Load your data
dat <- read.csv("your_data.csv")

# Set your API Key
api_key <- "your-api-key-here"
Sys.setenv(OPENAI_API_KEY = api_key)
```

**Note.** While it is optional to store your API Key in a variable, it is highly recommended to ensure secure storage and ease in programming. Alternatively, the researcher can set the API Key directly using the `Sys.setenv()` function.

## Define the Payload

Once the environment is set, it is now time to define a prompt and extract the LLMs response. The following example uses OpenAI's GPT-4. If researchers wish to use different LLMs, they can do so easily by replacing the link in the `url` variable with the appropriate one and adapt the model name in the function. Let us assume the dataset includes a list of survey responses regarding a government policy (stored in the `dat` variable) and the researcher wishes to classify them as either 'positive' or 'negative' sentiment.


```{r, echo=TRUE, eval=FALSE,linewidth=40}
# Define a function to annotate a sentence
annotate_sentence <- function(dat, api_key) {
    url <- "https://api.openai.com/v1/chat/completions"
    
    headers <- c(
        "Authorization" = paste("Bearer", api_key),
        "Content-Type" = "application/json"
    )
    
    body <- toJSON(list(
      model = "gpt-4",
      messages = list(
        list(role = "system", 
         content = "You are a research assistant"),
        list(role = "user", 
         content = paste(
           "Classify the following as 'positive' or 'negative' sentiment:", 
                            dat))
        ),
        max_tokens = 50
    ), auto_unbox = TRUE)
    
    response <- POST(url, add_headers(.headers = headers), body = body, 
                     encode = "json")
    content <- content(response, "parsed")
    
    
    # Extract the assistant's reply
    classification <- content$choices[[1]]$message$content
}

# Apply the classification function to each sentence
annotations <- sapply(dat, annotate_sentence, api_key = api_key)
```

**Note.** Listing a system role is highly advisable as it leads to more accurate responses. For efficiency, it is suggested that the researcher uses GPT-4-turbo. However, it is entirely the researcher's choice based on requirements and resources. Depending on the model the user chooses, the LLMs' pricing may vary.

## Error-Handling and Troubleshooting

Ensure that errors are handled properly. In certain cases, the LLM's response may not be exactly what the researcher expects. For instance, the response may include whitespaces, punctuations, or (more frustratingly) `NULL`. Therefore, it is acceptable to standardise them. This can be done in the following way:

```{r, echo=TRUE, results='hide', warning=FALSE, message=FALSE, eval=FALSE}
# Function to clean the classification
standardize_classification <- function(classification) {
    # Convert to lowercase, trim whitespace and remove punctuations
    classification <- tolower(trimws(classification))
    classification <- gsub("\\.$", "", classification)
    classification <- gsub("[[:punct:]]$", "", classification)
    # Define the valid responses
    if (classification %in% c("positive", "positive ")) {
        return("positive")
    } else if (classification %in% c("negative", "negative ")) {
        return("not a pledge")
    } else {
        return(NA_character_)  # Ensure it always returns a character string
    }
}
```

# 2. Using the `openai` R Package

The `openai` package simplifies the process of interacting with OpenAI's API and is compatible with the GPT-3 model and beyond. However, it is not possible to use any other Remote APIs with this package. If the researcher is decided on using OpenAI's API, this package may be helpful.

## Installation and Setup

The `openai` package can be installed using:

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE}
# Install Package
install.packages("openai")
```

Once installed, it requires a similar setup to the `httr` with `jsonlite` method.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
# Load the R Package
library(openai)

# Load your data
dat <- read.csv("your_data.csv")

# Set up API Key
api_key <- "your-api-key-here"
Sys.setenv(OPENAI_API_KEY = api_key)
```

## Defining the Prompt and Receiving Responses

While similar to the way we interact with ChatGPT via `httr` and `jsonlite`, the `openai` offers a bit more simplicity.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE, tidy=TRUE}
# Create a prompt
prompt <- paste("Classify the following as 'positive' or 'negative' sentiment", dat)

# Send the prompt to OpenAI and receive a response
response <- create_chat_completion(
  model = "gpt-4",
  prompt = prompt,
  max_tokens = 50
)

# Extract and print the response
classification <- response$choices[[1]]$text
```

For additional features and information, read the R package wrap [here](https://cran.r-project.org/web/packages/openai/openai.pdf).

# 3. Using the `chatgpt` R Package

The `chatgpt` package is another R-based package to interface OpenAI APIs. Using this package is the easiest way to interact with ChatGPT and it includes some high-level functions. The `chatgpt` package is designed specifically to make prompting OpenAI's API easy. Similar to the `openai` package though, it does not allow researchers to use any other Remote APIs. 

## Setting Up

The `chatgpt` package can be installed in the following way:

```{r, echo=TRUE, message=FALSE, results='hide', warning=FALSE, eval=FALSE}
install.packages("chatgpt")
```

Now, the next step is, as usual, loading the package and setting up the API Key.

```{r, echo=TRUE, message=FALSE, results='hide', warning=FALSE, eval=FALSE}
# Load the R Package
library(chatgpt)

# Load your data
dat <- read.csv("your_data.csv")

# Set up API Key
api_key <- "your-api-key-here"
Sys.setenv(OPENAI_API_KEY = api_key)
```

## Querying LLMs with `chatgpt`

While the `chatgpt` R package offers more flexibility than the `openai` package, it is not designed for heavy customisations. 

```{r, echo=TRUE, message=FALSE, results='hide', warning=FALSE, eval=FALSE, tidy=TRUE, tidy.opts=list(width.cutoff=50)}
# Create a function
annotate_sentence<- function(dat, 
                          model = "gpt-4", 
                          temperature = 0, 
                          max_tokens = 50) {
  prompt <- paste("Classify the following as 'positive' or 'negative' sentiment:", 
                  dat)
  response <- ask_chatgpt(prompt = prompt,
                          model = model,
                          temperature = temperature,
                          max_tokens = max_tokens)
  
  return(response)
}

# Apply the function to the text column
dat$sentiment <- sapply(dat$text, annotate_sentence)

# View results
head(dat)
```

# 4. Querying Local LLMs with R: the `httr` and `jsonlite` method

Local LLMs are increasingly popular, especially in research areas dealing with sensitive or confidential information. This is because Local LLMs are fully-customisable, offer offline use and cost-efficiency. However, this method requires more resources at the researcher's dispense for setting up. Once you have your own Local LLM set up, you can pass queries to it in similar fashion to the `httr` with `jsonlite` method. Please note, this handbook only concerns itself with how to query LLMs with R. To understand how to set up your own Local LLM, please use different resources.

## Setting Up

Local LLMs are usually served through a local REST API. For this example, a Local LLM is run using Ollama. This step must be edited as per the researcher's own requirement. If the researcher prefers to use ollama, it can be downloaded [here](https://ollama.com/). Importantly, this specific chunk is not R code, but should run on a computer's CLI, i.e. Terminal app for MacOS, Command Prompt for Windows, etc.

```{bash, eval=FALSE, echo=TRUE}
ollama run llama3 # Please edit this step as per your research requirement
```

## Installing R Packages

As mentioned above, this method is similar to the `httr` with `jsonlite` method. In fact, it uses the same packages to send/receive data.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE}
# Install Packages
install.packages("httr")
install.packages("jsonlite")
```

## Prepare and Send Query via R

By editing the `model_url` and `model` in the code below as per requirement, a researcher should typically be able to query Local LLMs via R.

```{r, eval=FALSE, echo=TRUE}
# Define a function to annotate a sentence
annotate_sentence <- function(dat, 
                              model_url = "http://localhost:11434/api/generate") 
  {
    body <- toJSON(list(
      model = "llama3",
      messages = list(
        list(role = "system", 
         content = "You are a research assistant"),
        list(role = "user", 
         content = paste(
           "Classify the following as 'positive' or 'negative' sentiment:", 
                            dat))
        ),
        max_tokens = 50
    ), auto_unbox = TRUE)
    
    response <- POST(model_url, add_headers(.headers = headers), body = body, 
                     encode = "json")
    content <- content(response, "parsed")
    
    
    # Extract the assistant's reply
    classification <- content$choices[[1]]$message$content
}

# Apply the classification function to each sentence
annotations <- sapply(dat, annotate_sentence, model_url)
```

Local LLMs are gaining traction and will continue to do so with the decrease in setup cost of LLMs and tightening data protection laws. Therefore, learning how to query Local LLMs can prove highly useful for researchers in the future.

# 5. Querying Local LLMs with `rollama`: Wrapper for Ollama API

The `rollama` package acts as a wrapper for the Ollama API and allows researchers to create a similar experience to OpenAI's API with Local LLMs. It is essentially a client package which requires `ollama` installed on the researcher's system separately for the model to run. As mentioned earlier, Ollama can simply be downloaded and installed from their website [here](https://ollama.com/). 

## Installation

The `rollama` package can be installed from CRAN by using:

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
install.packages("rollama")
```

The development version is also available via:

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
install.packages("remotes")
remotes::install_github("JBGruber/rollama")
```

Once you have `ollama` installed, you can verify your access to it with:

```{r, eval=FALSE, echo=TRUE, message=FALSE, warning=FALSE, results='hide'}
rollama::ping_ollama()
```

Now, call the `rollama` package and pull one of the models from the `ollama` library. A list of available models can be found [here](https://ollama.com/library). By using the `pull_model()` function, you can pull any of the `ollama` models. Without any arguments, the function pulls the default model - 'llama3.1 8b'.

```{r, eval=FALSE, echo=TRUE, message=FALSE, warning=FALSE, results='hide'}
# Install rollama package
library(rollama)

# Pull model from ollama
pull_model() # Without arguments, default model 'llama3.1 8b' called
```

## Making Queries

Structured queries can be created using the `make_query` function for tasks such as annotation of text. For a detailed explanation of available options, check the webpage [here](https://jbgruber.github.io/rollama/articles/annotation.html#the-make_query-helper-function).

```{r, echo=TRUE, message=FALSE, results='hide', warning=FALSE, eval=FALSE, tidy=TRUE, tidy.opts=list(width.cutoff=35)}
# Load your data
dat <- read.csv("your_data.csv")

# Create a query using make_query
q_zs <- make_query(
  text = dat$text,
  prompt = "Classify the following as 'positive' or 'negative' sentiment:",
  system = "You are a Research Assistant"
)
# Print the query
print(q_zs)

# Run the query using query() or chat()
query(q_zs, output = "text") # You can choose output type as "text", "list", "data.frame", 
        # "response", "httr2_response", "httr2_request".
```

For additional features such as model parameters and configuration settings relating to `rollama`, please check the documentation available [here](https://jbgruber.github.io/rollama/).

# 6. The `ellmer` R package

The `ellmer` package simplifies the use of LLMs via R. It supports a wide array of LLM providers, including OpenAI's API and Ollama. It also includes a rich set of functions for various tasks such as streaming outputs, tool/function calling, structured data extraction, etc. 

The key benefit of using this package is that it allows the researcher to choose between a programmatic chat or interactive chat console. The interactive chat console method makes it considerably easier for researchers, who are perhaps new to programming, to query LLMs.

## Installation

The `ellmer` R package can be downloaded from CRAN with:

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
install.packages("ellmer")
```

## List of Providers

A wide variety of LLM providers are supported via `ellmer`. Depending on the model, the setup is slightly different. For example, the OpenAI model requires you to obtain an API Key.

The supported models and their respective functions via the `ellmer` package are:

- Anthropic’s Claude: `chat_anthropic()`
- AWS Bedrock: `chat_aws_bedrock()`
- Azure OpenAI: `chat_azure_openai()`
- Cloudflare: `chat_cloudflare()`
- Databricks: `chat_databricks()`
- DeepSeek: `chat_deepseek()`
- GitHub model marketplace: `chat_github()`
- Google Gemini/Vertex AI: `chat_google_gemini()`, `chat_google_vertex()`
- Groq: `chat_groq()`
- Hugging Face: `chat_huggingface()`
- Mistral: `chat_mistral()`
- Ollama: `chat_ollama()`
- OpenAI: `chat_openai()`
- OpenRouter: `chat_openrouter()`
- perplexity.ai: `chat_perplexity()`
- Snowflake Cortex: `chat_snowflake()` and `chat_cortex_analyst()`
- VLLM: `chat_vllm()`

The choice of model to be used depends on the researcher's preference and requirements. It is highly recommended that the pros and cons of using the model are duly weighed out. For instance, Google Gemini is a free tier model, however, the data is used to improve the model. Hence, it will not be the best choice for sensitive data.

## Using `ellmer`

As mentioned earlier, the `ellmer` package supports a wide variety of LLMs. Therefore, there are several different ways to use the package. For this example, we use the OpenAI model. Whether the researcher wishes to use `ellmer` interactively or programmatically, both methods start with creating a new chat object.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
# Load the ellmer package
library(ellmer)

# Create a chat function
chat <- chat_openai(
  model = "gpt-4",
  api_key = "your-api-key-here",
  system_prompt = "You are a Research Assistant.",
)
```

Once the chat function is created, the researcher has the option of working on an interactive chat console or a programmatic chat.

**Note.** Irrespective of the chat function chosen, the chat object retains state. In other words, any previous interactions with the chat object are still part of the conversation. Any interactions the researcher has in the chat console will persist after exiting back to the R prompt.

### Interactive Chat Console

The most interactive method of using `ellmer` is to chat directly in your R console or browser with `live_console(chat)` or `live_browser`:

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
live_console(chat)
```

### Programmatic Chat

The programmatic chat option incorporates the chat object within a function. Then `$chat()` returns the result as a string.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
my_function <- function() {
  chat <- chat_openai(
    model = "gpt-4",
    api_key = "your-api-key-here"
    system_prompt = "You are a Research Assistant",
  )
  chat$chat("What is left and what is right on the political spectrum?")
}
my_function()
```

# 7. LLMs with Quanteda

The `quanteda.llm` package aims to simplify querying LLMs with data prepared with the `quanteda` R package, i.e. quanteda.corpora (or data frames). While still in its development stage, the `quanteda.llm` package allows the researcher to link the data to LLMs for analysing or classifying texts by creating new variables for what is created by the LLMs. In addition, it uses a tidy approach with the new `quanteda.tidy` package which enables convenient operations with common Tidyverse functions to manipulate LLM-created objects and variables.

The `quanteda.llm` package is compatible with all LLMs supported by the `ellmer` package. For authentication and usage of each LLM supported by `ellmer`, its original documentation is available [here](https://ellmer.tidyverse.org/).

## Set-Up - Development Version

To understand how to process the data with `quanteda` and installing necessary packages, check out the Quanteda tutorial [here](https://tutorials.quanteda.io/).

As the `quanteda.llm` package is not yet available on CRAN, the researcher may install its development version via:

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
install.packages("quanteda")
devtools::install_github("quanteda/quanteda.tidy")
install.packages("pak")
pak::pak("quanteda/quanteda.llm")
```

## Querying LLMs with Quanteda

Currently, the `quanteda.llm` package includes functions for specific tasks. As the package is developed further, it will support more functions for the researcher to exploit. Following is a list of all currently available functions with examples.

### Using `ai_summarize`

This function helps the researcher summarise documents in a corpus. The summary of the documents is then created as a new document variable in the corpus.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
library(quanteda)
library(quanteda.llm)
library(quanteda.tidy)
corpus <- quanteda::data_corpus_inaugural %>%
  quanteda.tidy::mutate(llm_sum = ai_summarize(text, 
                                               chat_fn = chat_ollama, 
                                               model = "llama3.2"))
# llm_sum is created as a new docvar in the corpus
```

### Using `ai_label`

The function assists in labelling documents within a corpus according to a content analysis scheme.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
library(quanteda)
library(quanteda.llm)
library(quanteda.tidy)
label = "Label the following document based on how much it aligns 
         with the political left, center, or right. 
         The political left is defined as groups which advocate for 
         social equality, government intervention in the economy, 
         and progressive policies.
         The political center typically supports a balance between 
         progressive and conservative views, 
         favoring moderate policies and compromise. 
         The political right generally advocates for individualism,
         free-market capitalism, and traditional values."
corpus <- quanteda::data_corpus_inaugural %>%
  quanteda.tidy::mutate(llm_label = ai_label(text, 
                                             chat_fn = chat_ollama, 
                                             model = "llama3.2", 
                                             label = label))
# llm_label is created as a new docvar in the corpus
```

### Using `ai_score`

Based on a defined scale provided by the researcher, this function helps in scoring the documents.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
library(quanteda)
library(quanteda.llm)
library(quanteda.tidy)
scale = "Score the following document on a scale of how much it aligns
         with the political left. The political left is defined as groups which 
         advocate for social equality, government intervention in the economy, 
         and progressive policies. Use the following metrics: 
         SCORING METRIC:
         1 : extremely left
         0 : not at all left"
corpus <- quanteda::data_corpus_inaugural %>%
  quanteda.tidy::mutate(llm_score = ai_score(text, 
                                             chat_fn = chat_ollama, 
                                             model = "llama3.2", 
                                             scale = scale))
# llm_score is created as a new docvar in the corpus
```

### Using `ai_validate`

The functions starts an interactive app to manually validate the LLM-generated content, i.e. summaries, labels or scores.

```{r, echo=TRUE, results='hide', message=FALSE, warning=FALSE, eval=FALSE}
library(quanteda)
library(quanteda.llm)
library(quanteda.tidy)
scale = "Score the following document on a scale of how much it aligns
         with the political left. The political left is defined as groups which 
         advocate for social equality, government intervention in the economy, 
         and progressive policies. Use the following metrics: 
         SCORING METRIC:
         1 : extremely left
         0 : not at all left"
corpus <- quanteda::data_corpus_inaugural %>%
  quanteda.tidy::mutate(llm_score = ai_score(text, 
                                             chat_fn = chat_ollama, 
                                             model = "llama3.2", 
                                             scale = scale))
# llm_score is created as a new docvar in the corpus
# Start the interactive app to validate the LLM-generated scores
corpus <- corpus %>%
  quanteda.tidy::mutate(validated = ai_validate(text, llm_score))
# validated is created as a new docvar in the corpus with all 
# non-validated scores set to NA
```

The `quanteda.llm` R package offers exciting functions already and will offer even more after its development stage. Stay tuned.

# Considerations

In the sections above, we discuss different R packages that can be used to query LLMs for research. With LLMs proving to be an incredible tool for researchers, it is important to understand which method best suits the research design and resources. Following is a list of considerations a researcher must have in mind when querying LLMs via R.

**1. Accuracy:**

- Validate the LLM responses against human annotations or traditional methods.
- There exists a possibility of error or misclassification by the LLM.
- Remote APIs, especially with their latest models, generally offer higher accuracy than Local LLMs which are dependent on the model's specifications for accuracy.

**2. Cost:**

- While LLMs are extremely useful, they are not cost-free.
- Remote APIs, such as OpenAI's API, incur costs per token. A token is roughly 4 characters. For large-scale jobs, consider Local LLMs.
- To ensure efficiency in cost, batch requests where possible and ensure proper debugging of code.
- While Local LLMs are free once set up, the setup cost itself can be high depending on the researcher's requirement. Budget accordingly.
  
**3. Data Privacy:**

- For sensitive or confidential data, consider using Local LLMs.
- Avoid sending sensitive data to Remote APIs.
  
**4. Reproducibility:**

- Different model versions, prompts and parameters can alter results. Researcher must ensure they are logged properly.
- Avoid stochastic generation unless analysing variance in responses.

**5. API Key Protection:**

- Keep your API Key protected and avoid deploying them on client-side environments.
- Sharing API Keys could be in breach of the API's Terms of Use.

# References

**Softwares**

- [R Installation](https://cran.r-project.org/)
- [Ollama Installation](https://ollama.com/)

**R Package Wrappers**

- [`httr` Package](https://cran.r-project.org/web/packages/httr/httr.pdf)
- [`jsonlite` Package](https://cran.r-project.org/web/packages/jsonlite/jsonlite.pdf)
- [`openai` Package](https://cran.r-project.org/web/packages/openai/openai.pdf)
- [`chatgpt` Package](https://cran.r-project.org/web/packages/chatgpt/chatgpt.pdf)
- [`rollama` Package - CRAN](https://cran.r-project.org/web/packages/rollama/rollama.pdf)
- [`rollama` Package - Github Page](https://jbgruber.github.io/rollama/)
- [`ellmer` Package - CRAN](https://cran.r-project.org/web/packages/ellmer/ellmer.pdf)
- [`ellmer` Package - Posit](https://ellmer.tidyverse.org/)
- [`quanteda.llm` Package - GitHub](https://github.com/quanteda/quanteda.llm?tab=readme-ov-file#supported-llms)

**Recommended Pages**

- [API Key Best Practices](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
- [Exploring LLMs in R](https://rpubs.com/shilad/llms_for_ds_f24_v2)
