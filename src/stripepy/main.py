import logging

from .cli import call, download, setup


def _setup_logger(level: str):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger().setLevel(level)


def main():
    subcommand, args = setup.parse_args()
    _setup_logger("INFO")  # TODO make tunable

    if subcommand == "call":
        return call.run(**args)
    if subcommand == "download":
        return download.run(**args)

    raise NotImplementedError


if __name__ == "__main__":
    main()
