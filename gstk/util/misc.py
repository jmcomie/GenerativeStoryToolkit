from datetime import datetime
from typing import Callable

import pytz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel


def format_unix_timestamp_with_timezone(epoch_timestamp, timezone_str="UTC") -> str:
    # Convert Unix timestamp to a datetime object
    dt_object = datetime.utcfromtimestamp(epoch_timestamp)

    # Make the datetime object timezone-aware
    timezone = pytz.timezone(timezone_str)
    aware_dt_object = timezone.localize(dt_object)

    # Format it to a string with the timezone
    formatted_time = aware_dt_object.strftime("%Y-%m-%d %H:%M:%S %Z")

    return formatted_time


# This is the output of a creation context function.
# In other words, these values are provided expressly, with the exception
# of the id, by the LLM.
def truncate_output_string_field_tokens(
    instance: BaseModel, token_count, length_function: Callable = len, exclude_fields: list[str] = []
):
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=token_count,
        chunk_overlap=0,
        length_function=length_function,
        add_start_index=True,
    )
    for field_name, value in instance.model_dump().items():
        if field_name in exclude_fields or not isinstance(value, str):
            continue
        if length_function(value) <= token_count:
            continue
        setattr(instance, field_name, text_splitter.create_documents([getattr(instance, field_name)])[0].page_content)
