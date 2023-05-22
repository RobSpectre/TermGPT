import os
import sys
import time
import random
import shutil
import textwrap

import click

from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferWindowMemory

@click.group()
def cli():
    pass


@cli.command(help="Chat with OpenAI")
@click.option("--system_message",
            type=str,
            default="Assistant is a large language model trained by OpenAI",
            help="Specify the system message for OpenAI")
@click.option("--model",
            type=str,
            default="gpt-3.5-turbo",
            help="Specify the OpenAI model to use")
@click.option("--temperature",
            type=int,
            default=0.5,
            help="Specify the temperature of the OpenAI model")
@click.option("--speed",
              type=int,
              default=130,
              help="Specify speed of the output.")
@click.option("--clear",
              is_flag=True,
              default=False,
              help="Clear terminal before presenting prompt.")
@click.option("--debug",
              is_flag=True,
              default=False,
              help="Turn on more verbose output.")
def chat(system_message, model, temperature, speed, clear, debug):
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')

    system_message_prompt = \
            SystemMessagePromptTemplate.from_template(system_message)
    human_message_prompt = \
        HumanMessagePromptTemplate.from_template("{input}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,
                                                    MessagesPlaceholder(variable_name='history'),
                                                    human_message_prompt])

    llm = ChatOpenAI(
        temperature=temperature,
        openai_api_key=os.environ.get("OPENAI_API_KEY", None),
        model_name=model
    )

    memory = ConversationBufferWindowMemory(return_messages=True)

    chain = ConversationChain(
        llm=llm,
        prompt=chat_prompt,
        memory=memory,
        verbose=debug
    )

    terminal_size = shutil.get_terminal_size()
    terminal_width = terminal_size.columns

    if debug:
        click.secho(f"Terminal width: {terminal_width}", fg='green')

    while True:
        question = click.prompt("$>", type=str)

        output = chain.predict(input=question)

        wrapped = textwrap.fill(output, width=terminal_width)
        
        for line in wrapped:
            for letter in line:
                sys.stdout.write(f"\033[33m{letter}\033[0m")
                sys.stdout.flush()
                time.sleep(random.random() * 10.0 / speed)

        print("\n")

if __name__ == "__main__":
    cli()
