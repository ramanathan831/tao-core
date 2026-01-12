#!/usr/bin/env python3
"""Update metadata for a skipped build."""

import json
import sys
import argparse


def update_skipped_metadata(input_file, output_file, build_info):
    """Update metadata to reflect current skipped build.
    
    Args:
        input_file: Previous build's metadata JSON
        output_file: Output file for updated metadata
        build_info: Dictionary with current build information
    """
    try:
        # Load previous metadata
        with open(input_file, 'r') as f:
            metadata = json.load(f)
        
        # Update build info for current skipped build
        metadata['build_info']['timestamp'] = build_info['timestamp']
        metadata['build_info']['build_number'] = build_info['build_number']
        metadata['build_info']['build_url'] = build_info['build_url']
        
        # Update build parameters
        metadata['build_parameters']['triggered_by'] = build_info['triggered_by']
        metadata['build_parameters']['skip_prod_promote'] = build_info['skip_prod_promote'] == 'True'
        
        # Update build status to reflect skip
        metadata['build_status']['skipped_build'] = True
        metadata['build_status']['duration_ms'] = build_info['duration_ms']
        metadata['build_status']['previous_build_reused'] = build_info['previous_build_number']
        metadata['build_status']['previous_successful_build'] = build_info['previous_build_url']
        
        # Write updated metadata
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Updated metadata for skipped build")
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update metadata for skipped build')
    parser.add_argument('--input', required=True, help='Previous metadata JSON file')
    parser.add_argument('--output', required=True, help='Output metadata JSON file')
    parser.add_argument('--timestamp', required=True, help='Current build timestamp')
    parser.add_argument('--build-number', required=True, help='Current build number')
    parser.add_argument('--build-url', required=True, help='Current build URL')
    parser.add_argument('--triggered-by', required=True, help='Who triggered the build')
    parser.add_argument('--skip-prod-promote', required=True, help='Skip prod promote flag')
    parser.add_argument('--duration-ms', required=True, type=int, help='Build duration in ms')
    parser.add_argument('--previous-build-number', required=True, help='Previous build number')
    parser.add_argument('--previous-build-url', required=True, help='Previous build URL')
    
    args = parser.parse_args()
    
    build_info = {
        'timestamp': args.timestamp,
        'build_number': args.build_number,
        'build_url': args.build_url,
        'triggered_by': args.triggered_by,
        'skip_prod_promote': args.skip_prod_promote,
        'duration_ms': args.duration_ms,
        'previous_build_number': args.previous_build_number,
        'previous_build_url': args.previous_build_url
    }
    
    update_skipped_metadata(args.input, args.output, build_info)


