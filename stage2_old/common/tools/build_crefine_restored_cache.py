import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stage2_old.common.tools.generate_coarse_cache import main


if __name__ == "__main__":
    main()
