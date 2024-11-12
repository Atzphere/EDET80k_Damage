import pygame
import sys
import math
import csv

# Initialize Pygame
pygame.init()

# Load the high-resolution image
image_path = '50x control ROI.png'  # Replace with your image path
original_image = pygame.image.load(image_path)

# Set the initial display size
display_width, display_height = 800, 600
screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Spot Measurement with Zoom and Pan")

# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Font for displaying information
font = pygame.font.Font(None, 36)

# Set up initial zoom and pan variables
scale = 1.0  # Initial scale (no zoom)
min_scale, max_scale = 0.1, 5.0  # Limits for zooming in and out
offset_x, offset_y = 0, 0  # Initial pan offsets
is_panning = False  # Flag to check if we are panning
pan_start = (0, 0)  # Track starting point of pan

# Data for measurements
current_points = []     # Temporarily holds two points for a measurement
measurements = []       # Stores tuples of (p1, p2, distance)

# Function to display instructions and count on the screen
def draw_instructions_and_count():
    instructions = font.render("Left-click to set points, F to save, Z to undo", True, WHITE)
    count_text = font.render(f"Measurements: {len(measurements)}", True, WHITE)
    screen.blit(instructions, (10, 10))
    screen.blit(count_text, (10, 40))

# Function to save measurements to CSV
def save_measurements_to_csv():
    with open('measurements.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Measurement Length'])
        for _, _, distance in measurements:
            writer.writerow([distance])
    print("Measurements saved to measurements.csv")

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click to add a point
                # Adjust for offset and scale to get original image coordinates
                x, y = event.pos
                original_x = int((x - offset_x) / scale)
                original_y = int((y - offset_y) / scale)
                
                # Store the point
                current_points.append((original_x, original_y))
                
                # If two points have been selected, calculate distance and add measurement
                if len(current_points) == 2:
                    # Calculate distance between points
                    p1, p2 = current_points
                    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    measurements.append((p1, p2, distance))
                    
                    # Clear points to allow for the next measurement
                    current_points = []

            elif event.button == 3:  # Right click to start panning
                is_panning = True
                pan_start = event.pos
            elif event.button == 4:  # Mouse wheel up (zoom in)
                # Calculate new scale, limiting to max_scale
                new_scale = min(scale * 1.1, max_scale)
                
                # Adjust offset to zoom towards cursor
                mouse_x, mouse_y = event.pos
                offset_x = (offset_x - mouse_x) * (new_scale / scale) + mouse_x
                offset_y = (offset_y - mouse_y) * (new_scale / scale) + mouse_y
                
                # Apply the new scale
                scale = new_scale

            elif event.button == 5:  # Mouse wheel down (zoom out)
                # Calculate new scale, limiting to min_scale
                new_scale = max(scale / 1.1, min_scale)
                
                # Adjust offset to zoom towards cursor
                mouse_x, mouse_y = event.pos
                offset_x = (offset_x - mouse_x) * (new_scale / scale) + mouse_x
                offset_y = (offset_y - mouse_y) * (new_scale / scale) + mouse_y
                
                # Apply the new scale
                scale = new_scale

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:  # Right click released, stop panning
                is_panning = False

        elif event.type == pygame.MOUSEMOTION:
            if is_panning:  # Update offset while panning
                dx, dy = event.rel
                offset_x += dx
                offset_y += dy

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z:  # Undo last measurement with "Z" key
                if measurements:
                    measurements.pop()
            elif event.key == pygame.K_f:  # Save measurements to CSV with "F" key
                save_measurements_to_csv()

    # Clear screen
    screen.fill((0, 0, 0))

    # Dynamically scale the original image for current zoom level
    scaled_image = pygame.transform.scale(
        original_image, (int(original_image.get_width() * scale), int(original_image.get_height() * scale))
    )
    # Display the scaled image with panning
    screen.blit(scaled_image, (offset_x, offset_y))

    # Draw each measurement line
    for p1, p2, _ in measurements:
        scaled_p1 = (int(p1[0] * scale + offset_x), int(p1[1] * scale + offset_y))
        scaled_p2 = (int(p2[0] * scale + offset_x), int(p2[1] * scale + offset_y))
        pygame.draw.line(screen, RED, scaled_p1, scaled_p2, 2)

    # Draw current points as red circles
    for point in current_points:
        scaled_point = (int(point[0] * scale + offset_x), int(point[1] * scale + offset_y))
        pygame.draw.circle(screen, RED, scaled_point, 5)

    # Display instructions and measurement count
    draw_instructions_and_count()

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
