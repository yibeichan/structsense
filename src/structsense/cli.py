"""This module defines CLI commands for the PipePal application."""

import logging

import click

from .app import kickoff

logger = logging.getLogger(__name__)


@click.group()
@click.pass_context
def cli(ctx):
    """CLI commands for the Structsense Framework application"""
    pass


@cli.command()
@click.option(
    "--agentconfig",
    required=True,
    type=str,
    help=("Path to the agent configuration in YAML file format or dictionary"),
)
@click.option(
    "--taskconfig",
    required=True,
    type=str,
    help=("Path to the agent task configuration in YAML format or or dictionary"),
)
@click.option(
    "--embedderconfig",
    required=True,
    type=str,
    help=("Path to the embedding configuration in YAML format or or dictionary"),
)

@click.option(
    "--knowledgeconfig",
    required=False,
    type=str,
    help=(
        "Path to the configuration in YAML format or or dictionary that specify the search knowledge search key."
    ),
)
@click.option(
    "--source",
    required=True,
    help=("The source—whether a file (text or PDF), a folder, or a text string."),
)
@click.option(
    "--humaninloop",
    required=False,
    help=("Option to indicate whether to enable human in the loop, default True"),
)
@click.option(
    "--agent_feedback_config",
    required=False,
    help=("Option provide the configuration that defines whether to enable human loop for the agents, e.g., extractor agent."
          "By default, human loop for other agents is disabled except for HumanFeedbackAgent."
          "Note: The option humaninloop should be set to True for this to work."),
)

def extract(
    agentconfig, taskconfig, embedderconfig, knowledgeconfig, source,
        humaninloop, agent_feedback_config
):
    """Extract the terms along with sentence."""
    logger.info(
        f"Processing source: {source} with agent config: {agentconfig}, task config: {taskconfig}, embedderconfig: {embedderconfig}, knowledgeconfig: {knowledgeconfig}"
    )
    click.echo(
        f"Processing source: {source} with agent config: {agentconfig}, task config: {taskconfig}, embedderconfig: {embedderconfig} knowledgeconfig: {knowledgeconfig}"
    )
    result = kickoff(
        agentconfig=agentconfig,
        taskconfig=taskconfig,
        embedderconfig=embedderconfig,
        knowledgeconfig=knowledgeconfig,
        input_source=source,
        enable_human_feedback=humaninloop,
        agent_feedback_config=agent_feedback_config

    )

    click.echo("*"*100)
    click.echo("Result")
    click.echo(result)
    click.echo("*"*100)


if __name__ == "__main__":
    cli()
