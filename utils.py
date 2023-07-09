from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

def example_rows(df):
    print("Example of a sarcastic text")
    print(df[df["labels"]=="sarcastic"].iloc[0]["text"])
    print()
    print("Example of a non-sarcastic text")
    print(df[df["labels"]=="not-sarcastic"].iloc[0]["text"])
    
    
def sarcasm_simple_prompt() -> list:
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="You are a model that generates sarcastic and non-sarcastic texts."
        )
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["num_generations", "direction"],
            template="Generate {num_generations} {direction} texts. Ensure diversity in the generated texts."
        )
    )

    return [system_message, human_message]


def sarcasm_annotate_prompt() -> list:
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="You are a model that annotates sarcastic and non-sarcastic texts."
        )
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["text"],
            template="""Classify the following text as being sarcastic or non-sarcastic. Reply with 'Sarcastic' if it's sarcastic and 'Non-sarcastic' if it's non-sarcastic. 
            Text: {text}
            """
        )
    )
    return [system_message, human_message]

def sarcasm_grounded_prompt() -> list:
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="You are a model that generates sarcastic and non-sarcastic texts."
        )
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["text", "num_generations", "direction"],
            template="""Rewrite the following text {num_generations} times to make it {direction}. 
            Make as few changes as possible to the text and stay true to its underlying style. 
            Text: {text}
            """
        )
    )
    return [system_message, human_message]

def sarcasm_grounded_no_rewrite_prompt() -> list:
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="You are a model that generates sarcastic and non-sarcastic texts."
        )
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["text", "num_generations", "direction"],
            template="""Here is an example of a {direction} text. Write {num_generations} new similar examples that have same {direction} tone. 
            Text: {text}
            """
        )
    )
    return [system_message, human_message]

def sarcasm_taxonomy_creation() -> list:
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="You are a model that thinks about ways a text can be sarcastic and non-sarcastic texts."
        )
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["num_ideas", "direction"],
            template="""Come up with {num_ideas} ways a text can be {direction}.
            """
        )
    )
    return [system_message, human_message]

def sarcasm_taxonomy_generation() -> list:
    system_message = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="You are a model that generates sarcastic and non-sarcastic texts."
        )
    )
    human_message = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["taxonomy", "num_generations", "direction", "text"],
            template="""
            Here are the ways a text can be {direction}: 
            {taxonomy}

            Your task it to rewrite the following text {num_generations} times to make it {direction}.
            For each rewrite, select one of the ways and use it. 
            Make as few changes as possible to the text and stay true to its underlying style. 

            Text: {text}
            """
        )
    )
    return [system_message, human_message]