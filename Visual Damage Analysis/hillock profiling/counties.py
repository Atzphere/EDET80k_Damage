import pygame
import sys

# Initialize Pygame
pygame.init()

# Load the high-resolution image
image_path = '50x control ROI.png'  # Replace with your image path
original_image = pygame.image.load(image_path)

# Set the initial display size
display_width, display_height = 800, 600
screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("Spot Counter with High-Detail Zoom and Pan")

# Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Font for displaying the count
font = pygame.font.Font(None, 36)

# Set up initial zoom and pan variables
scale = 1.0  # Initial scale (no zoom)
min_scale, max_scale = 0.1, 5.0  # Limits for zooming in and out
offset_x, offset_y = 0, 0  # Initial pan offsets
is_panning = False  # Flag to check if we are panning
pan_start = (0, 0)  # Track starting point of pan

# List to store spots in original image coordinates
spots = []

# Function to display count on the screen
def draw_count(count):
    count_text = font.render(f"Count: {count}", True, WHITE)
    screen.blit(count_text, (10, 10))

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click to add a spot
                # Adjust for offset and scale to get original image coordinates
                x, y = event.pos
                original_x = int((x - offset_x) / scale)
                original_y = int((y - offset_y) / scale)
                spots.append((original_x, original_y))
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
            if event.key == pygame.K_z:  # Undo last spot with "Z" key
                if spots:
                    spots.pop()

    # Clear screen
    screen.fill((0, 0, 0))

    # Dynamically scale the original image for current zoom level
    scaled_image = pygame.transform.scale(
        original_image, (int(original_image.get_width() * scale), int(original_image.get_height() * scale))
    )
    # Display the scaled image with panning
    screen.blit(scaled_image, (offset_x, offset_y))

    # Draw each spot (scaled and adjusted for offset)
    for spot in spots:
        scaled_spot = (
            int(spot[0] * scale + offset_x),
            int(spot[1] * scale + offset_y)
        )
        pygame.draw.circle(screen, RED, scaled_spot, 5)

    # Display the count of spots
    draw_count(len(spots))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
