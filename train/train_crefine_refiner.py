import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stage2_old.crefine.train.train_crefine_refiner import main


if __name__ == "__main__":
    main()
