"""
This is the UI component to the multi agent framework appication
using chainlit and Apache eCharts

Written by: Aaron Ward - October 2023.
"""
import chainlit as cl
from pathlib import Path

from agent import MultiAgent
from data_dictionary import dictionary

output_folder = "/Users/award40/Desktop/example_output"
# Get the path of the script
data_relative_path = './data/synthetic_covid_dataset_20230828.csv'
data_path = str(Path(__file__).absolute().parent.joinpath(data_relative_path).resolve())
data_loc_context = f"\n\nHere is the path to data that you should import: {data_path}"
data_dict_context = f"\n\nHere is the data dictionary: {dictionary.PROMPT_STRING}"

MAX_ITER = 10
USER_NAME = "User"
USER_PROXY_NAME = "Code Runner Agent"
ASSISTANT_NAME = "Programmer Agent"
WELCOME_MESSAGE = f"""
Datascience Agent Team 👾

\n\n

Here is the path to the data available to you: `{data_path}`
"""

##########################################################

@cl.on_chat_start
async def setup_agent():
    await cl.Avatar(
        name=USER_NAME,
        url="https://api.dicebear.com/7.x/thumbs/svg?seed=Callie&flip=true&rotate=350",
    ).send()

    await cl.Avatar(
        name=USER_PROXY_NAME,
        url="https://api.dicebear.com/7.x/bottts-neutral/svg?seed=Kiki&backgroundColor=757575",
    ).send()

    await cl.Avatar(
        name=ASSISTANT_NAME,
        url="https://api.dicebear.com/7.x/bottts-neutral/svg?seed=Kiki&backgroundColor=757575",
    ).send()

    agent = MultiAgent(work_dir=output_folder)
    agent.clear_history()
    coding_assistant, coding_runner = agent.instiate_agents()

    cl.user_session.set('agent', agent)
    cl.user_session.set(ASSISTANT_NAME, coding_assistant)
    cl.user_session.set(USER_PROXY_NAME, coding_runner)
    
    await cl.Message(content=WELCOME_MESSAGE).send()

@cl.on_file_upload(accept=["text/plain"], max_files=3, max_size_mb=2)
async def upload_file(files: any):
    """
    Handle uploaded files.
    Example:
        [{
            "name": "example.txt",
            "content": b"File content as bytes",
            "type": "text/plain"
        }]
    """
    for file_data in files:
        file_name = file_data["name"]
        content = file_data["content"]
        # If want to show content Content: {content.decode('utf-8')}\n\n
        await cl.Message(content=f"Uploaded file: {file_name}\n").send()
        
        # Save the file locally
        with open(file_name, "wb") as file:
            file.write(content)


@cl.on_message
async def run_conversation(user_message: str):
    try:
        # check if user message changed
        if user_message == cl.user_session.get('user_message'):
            return
        
        user_message += data_loc_context + data_dict_context

        agent = cl.user_session.get("agent")
        assistant = cl.user_session.get(ASSISTANT_NAME)
        user_proxy = cl.user_session.get(USER_PROXY_NAME)
        
        cur_iter = 0
        while cur_iter < MAX_ITER:
            if len(assistant.chat_messages[user_proxy]) == 0 :
                print('initiating chat')
                user_proxy.initiate_chat(
                    assistant,
                    message=user_message,
                    config_list=agent.config_list
                )
            else:
                print('FOLLOW up message')
                # followup of the previous question
                user_proxy.send(
                    recipient=assistant,
                    message=user_message,
                )
            
            message_history = assistant.chat_messages[user_proxy]
            last_seen_message_index = cl.user_session.get('last_seen_message_index', 0)
            print(message_history)

            naming_dict = {
                "User" : "You",
                "user" : USER_PROXY_NAME,
                "assistant": ASSISTANT_NAME,
            }

            for message in message_history[last_seen_message_index+1:]:
                await cl.Message(author=naming_dict[message["role"]], content=message["content"].replace("TERMINATE", "")).send()
            cl.user_session.set('last_seen_message_index', len(message_history))

            cur_iter += 1
            return
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()