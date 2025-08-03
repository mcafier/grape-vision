# main.py
import argparse
import sys
import os
from pathlib import Path


# ============================================================================
#               ROBUST PATHING CONFIGURATION
# This block ensures that the 'src' directory is on the Python path,
# so that we can import from it, no matter how this script is run.
# ============================================================================
# Get the absolute path of the directory containing this script (main.py)
project_root = os.path.dirname(os.path.abspath(__file__))
# Add the project root to the system's path
sys.path.insert(0, project_root)
# ============================================================================

# We have to do this to avoid errors, known ultralytics issue
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# We anticipate our pipeline classes will be in a 'src' directory.
# This ensures the script can find them.
try:
    from src.pipelines import ImageAnalysisPipeline, TimeSeriesPipeline
except ImportError as e:
    print("Error: Could not import pipeline classes.")
    print(f"Details: {e}") # Adding the specific error message helps debugging
    print("Please ensure you are running this script from the project's root directory,")
    print("and that the 'src' directory with an '__init__.py' file exists.")
    sys.exit(1)

def main():
    """
    Main entry point for the Grape Vision Command-Line Interface (CLI).
    
    This function parses user commands and arguments, then instantiates and runs
    the appropriate analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="A complete computer vision pipeline to estimate grape berry volume and track them over time."
    )
    
    # Create subparsers for each main command
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="The main task to perform. Available commands: detect, analyze, track, track-analyze, analyze-crops"
    )

    # --- Sub-parser for the 'detect' command ---
    parser_detect = subparsers.add_parser(
        "detect", 
        help="Run ONLY detection on an image or folder. Saves bounding boxes and labels."
    )
    parser_detect.add_argument("-i", "--input", required=True, help="Path to the input image or folder.")
    parser_detect.add_argument("-o", "--output", default=None, help="Path to the output directory. Defaults to a 'Results' folder next to the input.")
    parser_detect.add_argument("-c", "--config", default="configs\config.yaml", help="Path to the main configuration file.")

    # --- Sub-parser for the 'analyze' command ---
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="Run the FULL analysis pipeline (detection + volume) on an image or folder."
    )
    parser_analyze.add_argument("-i", "--input", required=True, help="Path to the input image or folder.")
    parser_analyze.add_argument("-o", "--output", default=None, help="Path to the output directory. Defaults to a 'Results' folder next to the input.")
    parser_analyze.add_argument("-c", "--config", default="configs\config.yaml", help="Path to the main configuration file.")

    # --- Sub-parser for the 'track' command ---
    parser_track = subparsers.add_parser(
        "track",
        help="Run ONLY tracking on a sequence of images. Saves tracking data and optional crops."
    )
    parser_track.add_argument("-i", "--input", required=True, help="Path to the input folder of sequential images.")
    parser_track.add_argument("-o", "--output", default=None, help="Path to the output directory. Defaults to a 'Results' folder next to the input.")
    parser_track.add_argument("-c", "--config", default="configs\config.yaml", help="Path to the main configuration file.")

    # --- Sub-parser for the 'track-analyze' command ---
    parser_track_analyze = subparsers.add_parser(
        "track-analyze",
        help="Run the FULL time-series analysis (tracking + volume) on an image sequence."
    )
    parser_track_analyze.add_argument("-i", "--input", required=True, help="Path to the input folder of sequential images.")
    parser_track_analyze.add_argument("-o", "--output", default=None, help="Path to the output directory. Defaults to a 'Results' folder next to the input.")
    parser_track_analyze.add_argument("-c", "--config", default="configs\config.yaml", help="Path to the main configuration file.")
    
    # --- Sub-parser for the 'analyze-crops' command ---
    parser_analyze_crops = subparsers.add_parser(
        "analyze-crops",
        help="Run volume analysis ONLY on pre-tracked folders of cropped berries."
    )
    parser_analyze_crops.add_argument("-i", "--input", required=True, help="Path to the root folder containing tracked crop folders (e.g., 'Berry_1', 'Berry_2').")
    parser_analyze_crops.add_argument("-o", "--output", default=None, help="Path to the output directory. Defaults to a 'Results' folder next to the input.")
    parser_analyze_crops.add_argument("-c", "--config", default="configs\config.yaml", help="Path to the main configuration file.")

    args = parser.parse_args()

    # --- Dispatch to the correct pipeline based on the user's command ---
    print(f"Executing '{args.command}' command...")
    

    try:
        input_path = Path(args.input).resolve()
        # If output is not specified, determine default based on input
        if args.output is None:
            if input_path.is_dir():
                output_path = input_path / "Results" # pathlib's elegant way to join paths
            else:
                output_path = input_path.parent / "Results"
        else:
            output_path = Path(args.output)
    except FileNotFoundError:
        print(f"\nError: The specified input path does not exist: {args.input}")
        sys.exit(1)
        
        # Ensure the output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Executing '{args.command}' command...")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")

    # --- Dispatch to pipelines, passing the Path objects ---
    try:
        if args.command in ["detect", "analyze"]:
            pipeline = ImageAnalysisPipeline(config_path=args.config)
            if args.command == "detect":
                # The pipeline methods will receive Path objects but will work fine
                # because functions like os.path.join handle them correctly.
                pipeline.run_detection_only(input_path=input_path, output_dir=output_path)
            elif args.command == "analyze":
                pipeline.run_full_analysis(input_path=input_path, output_dir=output_path)
    
        elif args.command in ["track", "track-analyze", "analyze-crops"]:
            pipeline = TimeSeriesPipeline(config_path=args.config)
            if args.command == "track":
                pipeline.run_tracking_only(input_dir=input_path, output_dir=output_path)
            elif args.command == "track-analyze":
                pipeline.run_full_analysis(input_dir=input_path, output_dir=output_path)
            elif args.command == "analyze-crops":
                pipeline.run_analysis_on_crops(input_dir=input_path, output_dir=output_path)

        print(f"\nCommand '{args.command}' completed successfully.")
        print(f"Results saved to: {output_path.resolve()}")

    except FileNotFoundError as e:
        print(f"\nError: A file or directory was not found.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # For debugging, we might want to print the full traceback
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()