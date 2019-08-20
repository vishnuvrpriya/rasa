import asyncio
import os
import shutil
from typing import Text, List

import matplotlib
import pytest
import logging

from async_generator import yield_, async_generator
from sanic import Sanic

import rasa.utils.io
import rasa.nlu.config
import rasa.core.config
from rasa.core.policies import Policy
from rasa.core.domain import Domain
from rasa.constants import (
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODELS_PATH,
)
from rasa.core.interpreter import RegexInterpreter
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.run import _create_app_without_api
from rasa import server
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import RestInput
from rasa.core.channels import channel
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.model import get_model
from rasa.train import train_async


matplotlib.use("Agg")


# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.

pytest_plugins = ["pytester"]


DOMAIN_PATH_WITH_SLOTS = "data/test_domains/default_with_slots.yml"
DOMAIN_PATH_WITH_MAPPING = "data/test_domains/default_with_mapping.yml"

END_TO_END_STORY_FILE = "data/test_evaluations/end_to_end_story.md"
END_TO_END_STORY_FILE_UNKNOWN_ENTITY = "data/test_evaluations/story_unknown_entity.md"

MOODBOT_MODEL_DIRECTORY = "examples/moodbot/"


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)

    yield loop

    loop.close()


@pytest.fixture(autouse=True)
def set_log_level_debug(caplog):
    # Set the post-test log level to DEBUG for failing tests.  For all tests
    # (failing and successful), the live log level can be additionally set in
    # `setup.cfg`. It should be set to WARNING.
    caplog.set_level(logging.DEBUG)


@pytest.fixture(scope="session")
def project() -> Text:
    import tempfile
    from rasa.cli.scaffold import create_initial_project

    directory = tempfile.mkdtemp()
    create_initial_project(directory)

    yield directory

    if os.path.exists(directory):
        shutil.rmtree(directory)


###############
# DEFAULT FILES
###############


@pytest.fixture(scope="session")
def default_domain_path_with_slots() -> Text:
    return DOMAIN_PATH_WITH_SLOTS


@pytest.fixture(scope="session")
def default_domain_path_with_mapping() -> Text:
    return DOMAIN_PATH_WITH_MAPPING


@pytest.fixture(scope="session")
def default_domain_path(project: Text) -> Text:
    return os.path.join(project, DEFAULT_DOMAIN_PATH)


@pytest.fixture(scope="session")
def default_stories_file(project: Text) -> Text:
    return os.path.join(project, DEFAULT_DATA_PATH, "stories.md")


@pytest.fixture(scope="session")
def default_nlu_file(project: Text) -> Text:
    return os.path.join(project, DEFAULT_DATA_PATH, "nlu.md")


@pytest.fixture(scope="session")
def default_config_path(project: Text) -> Text:
    return os.path.join(project, DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="session")
def end_to_end_story_file() -> Text:
    return END_TO_END_STORY_FILE


@pytest.fixture(scope="session")
def end_to_end_story_file_with_unkown_entity() -> Text:
    return END_TO_END_STORY_FILE_UNKNOWN_ENTITY


@pytest.fixture(scope="session")
def default_core_config(default_config_path: Text) -> List[Policy]:
    return rasa.core.config.load(default_config_path)


@pytest.fixture(scope="session")
def default_nlu_config(default_config_path: Text) -> rasa.nlu.config.RasaNLUModelConfig:
    return rasa.nlu.config.load(default_config_path)


@pytest.fixture(scope="session")
def default_domain(default_domain_path) -> Domain:
    return Domain.load(default_domain_path)


#######
# AGENT
#######


@pytest.fixture
@async_generator
async def stack_agent(trained_model: Text) -> Agent:
    agent = await load_agent(model_path=trained_model)
    await yield_(agent)
    del agent


@pytest.fixture
@async_generator
async def core_agent(trained_core_model: Text) -> Agent:
    agent = await load_agent(model_path=trained_core_model)
    await yield_(agent)
    del agent


@pytest.fixture
@async_generator
async def nlu_agent(trained_nlu_model: Text) -> Agent:
    agent = await load_agent(model_path=trained_nlu_model)
    await yield_(agent)
    del agent


@pytest.fixture
@async_generator
async def default_agent(
    tmpdir_factory, default_stories_file: Text, default_domain: Domain
) -> Agent:
    agent = Agent(
        default_domain,
        policies=[MemoizationPolicy()],
        interpreter=RegexInterpreter(),
        tracker_store=InMemoryTrackerStore(default_domain),
    )

    training_data = await agent.load_data(default_stories_file)
    agent.train(training_data)

    await yield_(agent)
    del agent


@pytest.fixture
async def default_agent_path(default_agent: Agent, tmpdir_factory) -> Text:
    path = tmpdir_factory.mktemp("agent").strpath
    default_agent.persist(path)
    return path


#############
# RASA SERVER
#############


@pytest.fixture
@async_generator
async def rasa_server(stack_agent: Agent) -> Sanic:
    app = server.create_app(agent=stack_agent)
    channel.register([RestInput()], app, "/webhooks/")
    await yield_(app)
    del app


@pytest.fixture
@async_generator
async def rasa_core_server(core_agent: Agent) -> Sanic:
    app = server.create_app(agent=core_agent)
    channel.register([RestInput()], app, "/webhooks/")
    await yield_(app)
    del app


@pytest.fixture
@async_generator
async def rasa_nlu_server(nlu_agent: Agent) -> Sanic:
    app = server.create_app(agent=nlu_agent)
    channel.register([RestInput()], app, "/webhooks/")
    await yield_(app)
    del app


@pytest.fixture
@async_generator
async def rasa_server_secured(default_agent: Agent) -> Sanic:
    app = server.create_app(agent=default_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    await yield_(app)
    del app


@pytest.fixture
def rasa_server_without_api() -> Sanic:
    app = _create_app_without_api()
    channel.register([RestInput()], app, "/webhooks/")
    yield app
    del app


################
# TRAINED MODELS
################


@pytest.fixture
@async_generator
async def trained_model(project: Text) -> Text:
    model_path = await train_model(project)
    await yield_(model_path)
    shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture
@async_generator
async def trained_core_model(project: Text) -> Text:
    model_path = await train_model(project, model_type="core")
    await yield_(model_path)
    shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture
@async_generator
async def trained_nlu_model(project: Text) -> Text:
    model_path = await train_model(project, model_type="nlu")
    await yield_(model_path)
    shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture
@async_generator
async def trained_moodbot_path():
    model_path = await train_async(
        domain=os.path.join(MOODBOT_MODEL_DIRECTORY, DEFAULT_DOMAIN_PATH),
        config=os.path.join(MOODBOT_MODEL_DIRECTORY, DEFAULT_CONFIG_PATH),
        training_files=os.path.join(MOODBOT_MODEL_DIRECTORY, DEFAULT_DATA_PATH),
        output_path=os.path.join(MOODBOT_MODEL_DIRECTORY, DEFAULT_MODELS_PATH),
    )
    await yield_(model_path)
    shutil.rmtree(model_path, ignore_errors=True)


@pytest.fixture
def unpacked_trained_moodbot_path(trained_moodbot_path: Text):
    model_dir = get_model(trained_moodbot_path)
    yield model_dir
    shutil.rmtree(model_dir, ignore_errors=True)


async def train_model(
    project: Text, filename: Text = "test.tar.gz", model_type: Text = "stack"
):
    output = os.path.join(project, DEFAULT_MODELS_PATH, filename)
    domain = os.path.join(project, DEFAULT_DOMAIN_PATH)
    config = os.path.join(project, DEFAULT_CONFIG_PATH)

    if model_type == "core":
        training_files = os.path.join(project, DEFAULT_DATA_PATH, "stories.md")
    elif model_type == "nlu":
        training_files = os.path.join(project, DEFAULT_DATA_PATH, "nlu.md")
    else:
        training_files = os.path.join(project, DEFAULT_DATA_PATH)

    await train_async(domain, config, training_files, output)

    return output
