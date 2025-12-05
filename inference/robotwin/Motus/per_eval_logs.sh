#!/bin/bash

# Directory containing the log files

# NOTE: Change to your own log directory as needed
log_dir="/share/home/bhz/test/RoboTwin/logs_robotwin_stage4_pretrain_5e5_1113_40k"

if [ ! -d "$log_dir" ]; then
    echo "Log directory not found: $log_dir"
    exit 1
fi

echo "Parsing evaluation logs from: $log_dir"
echo "--------------------------------------------------"
echo "Task                               | Success Rate (%)"
echo "-----------------------------------|------------------"

total_success_rate=0
task_count=0
total_tasks_processed=0
total_successful_tasks=0
total_failed_tasks=0
error_count=0
not_found_count=0

# Find all log files, sorting them for consistent output
log_files=($(find "$log_dir" -name "*.log" | sort))

if [ ${#log_files[@]} -eq 0 ]; then
    echo "No log files with pattern '*.log' found in $log_dir"
    exit 1
fi

for file in "${log_files[@]}"; do
    # Extract success rate from the last 10 lines of the file. This is efficient for large files.
    # We take the last matching line, as that should be the final one (e.g., out of 100)
    success_line=$(tail -n 10 "$file" | grep "Success rate:" | tail -n 1)
    
    if [ -n "$success_line" ]; then
        # Use grep with Perl-compatible regex (-P) to extract the floating point number followed by a %
        # Then remove the trailing '%'
        success_percentage=$(echo "$success_line" | grep -oP '(\d+\.\d+)%' | sed 's/%//')

        if [[ "$success_percentage" =~ ^[0-9.]+$ ]]; then
            # Convert percentage to a decimal for calculation
            success_rate=$(echo "scale=6; $success_percentage / 100" | bc)
            
            # Extract task name from filename, e.g., "beat_block_hammer.log" -> "beat_block_hammer"
            filename=$(basename "$file")
            task_name=$(echo "$filename" | sed 's/\.log$//')
            
            # Print in a formatted table
            printf "%-35s| %.2f%%\n" "$task_name" "$success_percentage"
            
            # Accumulate for average calculation using bc for floating point math
            total_success_rate=$(echo "scale=6; $total_success_rate + $success_rate" | bc)
            task_count=$((task_count + 1))
            
            # Count successful and failed tasks based on success rate
            if (( $(echo "$success_percentage > 0" | bc -l) )); then
                total_successful_tasks=$((total_successful_tasks + 1))
            else
                total_failed_tasks=$((total_failed_tasks + 1))
            fi
            total_tasks_processed=$((total_tasks_processed + 1))
        else
            filename=$(basename "$file")
            task_name=$(echo "$filename" | sed 's/\.log$//')
            printf "%-35s| Error parsing value\n" "$task_name"
            error_count=$((error_count + 1))
        fi
    else
        filename=$(basename "$file")
        task_name=$(echo "$filename" | sed 's/\.log$//')
        printf "%-35s| Not found\n" "$task_name"
        not_found_count=$((not_found_count + 1))
    fi
done

echo "--------------------------------------------------"

if [ "$task_count" -gt 0 ]; then
    # Calculate and print the average success rate as a percentage
    average_percentage=$(echo "scale=4; ($total_success_rate / $task_count) * 100" | bc)
    # Round to 2 decimal places using printf
    average_percentage=$(printf "%.2f" "$average_percentage")
    echo "Processed $task_count tasks."
    printf "Average Success Rate: %s%%\n" "$average_percentage"
else
    echo "Could not find success rate in any log file."
fi