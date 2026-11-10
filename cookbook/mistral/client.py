import os

from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
