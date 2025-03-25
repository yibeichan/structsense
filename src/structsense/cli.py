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
    "--source",
    required=True,
    help=("The source—whether a file (text or PDF), a folder, or a text string."),
)
def extract(agentconfig, taskconfig, embedderconfig, source):
    """Extract the terms along with sentence."""
    logger.info(
        f"Processing source: {source} with agent config: {agentconfig} and task config: {taskconfig}"
    )
    click.echo(
        f"Processing source: {source} with agent config: {agentconfig} and task config: {taskconfig}"
    )
    result = kickoff(agentconfig, taskconfig, embedderconfig, source)
    click.echo(
        result)

if __name__ == "__main__":
    cli()
