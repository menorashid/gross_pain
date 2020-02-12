import pandas as pd
import numpy as np
import subprocess
import argparse
import sys
import ast

from multiview_frame_extractor import MultiViewFrameExtractor


def parse_arguments(argv):
  """Parses the arguments passed to the extract_frames.py script."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--output_dir', type=str,
      help='''Root directory where to save all frames.''')
  parser.add_argument('--csv_path', type=str,
      help='Path to .csv-file listing all the times to extract from.')
  parser.add_argument('--views', type=str,
      help='Which viewpoints to include in the dataset, e.g. all [0,1,2,3]')
  parser.add_argument('--width', type=int,
      help="Width of extracted frames.")
  parser.add_argument('--height', type=int,
      help="Height of extracted frames.")
  parser.add_argument('--frame_rate', type=int,
      help="Frame rate to extract at.")
  return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    views = ast.literal_eval(args.views)

    frame_extractor = MultiViewFrameExtractor(width=args.width,
                                              height=args.height,
                                              frame_rate=args.frame_rate,
                                              output_dir=args.output_dir,
                                              views=views,
                                              data_selection_path=args.csv_path)

    frame_extractor.create_clip_directories()
    frame.extractor.extract_frames()


