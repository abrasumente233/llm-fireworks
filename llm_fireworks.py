import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from pathlib import Path
import json
import time
import httpx

def get_fireworks_models():
    return fetch_cached_json(
        url="https://api.fireworks.ai/inference/v1/models",
        path=llm.user_dir() / "fireworks_models.json",
        cache_timeout=3600,
    )["data"]

class FireworksChat(Chat):
    needs_key = "fireworks"
    key_env_var = "LLM_FIREWORKS_KEY"
    def __str__(self):
        return "Fireworks: {}".format(self.model_id)

class FireworksAsyncChat(AsyncChat):
    needs_key = "fireworks"
    key_env_var = "LLM_FIREWORKS_KEY"
    def __str__(self):
        return "Fireworks: {}".format(self.model_id)

@llm.hookimpl
def register_models(register):
    # Only do this if the fireworks key is set
    key = llm.get_key("", "fireworks", "LLM_FIREWORKS_KEY")
    if not key:
        return
    for model_definition in get_fireworks_models():
        supports_images = get_supports_images(model_definition)
        kwargs = dict(
            model_id="fireworks/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            vision=supports_images,
            api_base="https://api.fireworks.ai/inference/v1",
            headers={"X-Title": "LLM"},
        )
        register(
            FireworksChat(**kwargs),
            FireworksAsyncChat(**kwargs),
        )

class DownloadError(Exception):
    pass

def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)
    # Create directories if not exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.is_file():
        # Get the file's modification time
        mod_time = path.stat().st_mtime
        # Check if it's more than the cache_timeout old
        if time.time() - mod_time < cache_timeout:
            # If not, load the file
            with open(path, "r") as file:
                return json.load(file)

    # Try to download the data
    try:
        key = llm.get_key("", "fireworks", "LLM_FIREWORKS_KEY")
        headers = {"Authorization": f"Bearer {key}"}
        response = httpx.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()  # This will raise an HTTPError if the request fails
        # If successful, write to the file
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        # If there's an existing file, load it
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            # If not, raise an error
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )

def get_supports_images(model_definition):
    try:
        # e.g. `text->text` or `text+image->text`
        modality = model_definition["architecture"]["modality"]
        input_modalities = modality.split("->")[0].split("+")
        return "image" in input_modalities
    except Exception:
        return False
