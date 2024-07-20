import yaml
import json


def dump_yml(data: dict, path: str) -> None:
    """Interface to read a yml file.

    Args:
        data (dict): data to dump
        path (str): path to target file
    """
    with open(path, "w") as yml_file:
        yaml.safe_dump(data, yml_file, sort_keys=False, default_flow_style=False)


def read_yml(path: str) -> dict:
    """Interface to read a yml file.

    Args:
        path (str): path to target file
    Returns:
        (dict) content of yml file
    """
    with open(path, "r") as yml_file:
        data = yaml.safe_load(yml_file)
    return data


def read_json(path: str) -> dict:
    """Interface to read a json file.

    Args:
        path (str): path to target file

    Returns:
        content of yml file
    """
    with open(path, "r") as fp:
        data = json.load(fp)

    return data


def dump_json(data: dict, path: str) -> None:
    """Interface to dump a json file.

    Args:
        data (dict): data to dump
        path (str): path to target file
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)
