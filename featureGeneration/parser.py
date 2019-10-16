"""Feature Extraction.

Usage: parser INPUT OUTPUT GEOMETRY ID N
"""
from docopt import docopt

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments["INPUT"])

