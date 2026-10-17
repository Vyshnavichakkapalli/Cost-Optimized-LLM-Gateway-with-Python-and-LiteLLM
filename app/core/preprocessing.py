import re


def preprocess_prompt(prompt: str) -> str:
    """
    Cleans and normalizes the prompt before routing, token estimation, or caching:
    - Removes unprintable characters (preserving standard control chars like newline/tab).
    - Normalizes line endings (\r\n -> \n, \r -> \n).
    - Reduces 3 or more consecutive newline characters to a maximum of 2 (\n\n).
    - Strips redundant trailing/leading whitespace per line or overall.
    - Strips leading and trailing whitespace from the final prompt.
    """
    if not prompt:
        return ""

    # 1. Filter unprintable characters while preserving standard whitespaces (\n, \t)
    filtered = "".join(ch for ch in prompt if ch in "\n\t\r" or ch.isprintable())

    # 2. Normalize Windows/Mac line endings to standard Unix \n
    filtered = filtered.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Reduce multiple consecutive newline characters (\n\n\n+) to a maximum of two (\n\n)
    filtered = re.sub(r"\n{3,}", "\n\n", filtered)

    # 4. Normalize trailing spaces on individual lines
    lines = [line.rstrip() for line in filtered.split("\n")]
    filtered = "\n".join(lines)

    # 5. Strip leading and trailing whitespace of the overall prompt
    return filtered.strip()
