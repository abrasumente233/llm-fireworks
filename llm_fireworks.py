import llm
from llm.default_plugins.openai_models import Chat
from pathlib import Path
import json
import time
import httpx

# Try to import AsyncChat, but don't fail if it's not available
try:
    from llm.default_plugins.openai_models import AsyncChat

    HAS_ASYNC = True
except ImportError:
    HAS_ASYNC = False


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


# Only define AsyncChat class if async support is available
if HAS_ASYNC:

    class FireworksAsyncChat(AsyncChat):
        needs_key = "fireworks"
        key_env_var = "LLM_FIREWORKS_KEY"

        def __str__(self):
            return "Fireworks: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
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

        if HAS_ASYNC:
            register(
                FireworksChat(**kwargs),
                FireworksAsyncChat(**kwargs),
            )
        else:
            register(FireworksChat(**kwargs))


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            with open(path, "r") as file:
                return json.load(file)

    try:
        key = llm.get_key("", "fireworks", "LLM_FIREWORKS_KEY")
        headers = {"Authorization": f"Bearer {key}"}
        response = httpx.get(url, headers=headers, follow_redirects=True)
        response.raise_for_status()
        with open(path, "w") as file:
            json.dump(response.json(), file)
        return response.json()
    except httpx.HTTPError:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )


def get_supports_images(model_definition):
    try:
        modality = model_definition["architecture"]["modality"]
        input_modalities = modality.split("->")[0].split("+")
        return "image" in input_modalities
    except Exception:
        return False
