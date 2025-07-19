"""
Grid rendering utilities for visualizing ARC problems
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..data import ARCProblem, ARCPair


# Color palette for ARC (0-9)
COLORS = [
    (0, 0, 0),       # 0: Black
    (0, 116, 217),   # 1: Blue
    (255, 65, 54),   # 2: Red
    (46, 204, 64),   # 3: Green
    (255, 220, 0),   # 4: Yellow
    (160, 160, 160), # 5: Gray
    (240, 18, 190),  # 6: Pink
    (255, 133, 27),  # 7: Orange
    (127, 219, 255), # 8: Purple (was Teal)
    (135, 12, 37),   # 9: Brown (was Maroon)
]


@dataclass
class RenderResult:
    """Result of rendering grids"""
    image_path: str
    image: Image.Image
    grid_info: dict  # Metadata about rendered grids


class GridRenderer:
    """Render ARC grids as images for GLM evaluation"""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = output_dir
        self.colors = COLORS
        
    def render_arc_problem(self, problem: ARCProblem, output_path: str) -> str:
        """
        Render ARC problem (training pairs only) in GLM-4.1V style
        """
        train_examples = []
        for pair in problem.train_pairs:
            train_examples.append({
                'input': pair.x.tolist(),
                'output': pair.y.tolist()
            })
        
        # Test input for reference but not included in this image
        problem_data = {
            'train': train_examples,
            'test': [{'input': problem.test_pairs[0].x.tolist()}] if problem.test_pairs else []
        }
        
        return self._render_arc_problem_glm_style(problem_data, output_path, include_test=False)
    
    def render_solution_result(self, 
                              test_input: np.ndarray, 
                              generated_output: np.ndarray,
                              output_path: str) -> str:
        """
        Render test input and generated output pair
        """
        padding = 20
        gap = 60
        uniform_grid_size = 240
        
        # Calculate optimal cell size
        all_grids = [test_input, generated_output]
        max_grid_h = max(len(grid) for grid in all_grids)
        max_grid_w = max(len(grid[0]) for grid in all_grids)
        
        # Calculate cell size with better minimum
        unified_cell_size = min(uniform_grid_size // max_grid_h, 
                               uniform_grid_size // max_grid_w)
        # Ensure minimum readable cell size
        unified_cell_size = max(unified_cell_size, 12)
        # Cap maximum cell size to prevent huge grids for small problems
        unified_cell_size = min(unified_cell_size, 25)
        
        # Layout: 2 grids side by side (test input -> generated output)
        total_width = 2 * uniform_grid_size + gap + 2 * padding
        total_height = uniform_grid_size + 50 + 2 * padding  # 50 for text
        
        # Create image
        img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw test input
        self._draw_grid_centered(draw, test_input,
                               padding, padding + 30,
                               uniform_grid_size, unified_cell_size)
        # Load font for labels
        try:
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            label_font = ImageFont.load_default()
            
        draw.text((padding, padding), "Test Input", fill='black', font=label_font)
        
        # Draw generated output
        output_x = padding + uniform_grid_size + gap
        self._draw_grid_centered(draw, generated_output,
                               output_x, padding + 30,
                               uniform_grid_size, unified_cell_size)
        draw.text((output_x, padding), "Generated Output", fill='green', font=label_font)
        
        img.save(output_path)
        return output_path
    
    def _render_arc_problem_glm_style(self, problem_data, output_path: str, include_test: bool = False) -> str:
        """Render ARC problem in GLM-4.1V style (adapted from original)"""
        train_examples = problem_data['train']
        test_input = problem_data['test'][0]['input'] if problem_data['test'] and include_test else None
        
        padding = 40
        gap = 100
        row_gap = 100
        uniform_grid_size = 250
        
        # Collect all grids for sizing
        all_grids = []
        for example in train_examples:
            all_grids.extend([example['input'], example['output']])
        if test_input is not None:
            all_grids.append(test_input)
        
        # Find optimal cell size
        max_grid_h = max(len(grid) for grid in all_grids)
        max_grid_w = max(len(grid[0]) for grid in all_grids)
        
        # Calculate cell size with better minimum
        unified_cell_size = min(uniform_grid_size // max_grid_h, 
                               uniform_grid_size // max_grid_w)
        # Ensure minimum readable cell size
        unified_cell_size = max(unified_cell_size, 12)
        # Cap maximum cell size to prevent huge grids for small problems
        unified_cell_size = min(unified_cell_size, 25)
        
        # Layout: 2 grids per row (input-output pairs), with test input on last row if included
        num_rows = len(train_examples) + (1 if test_input is not None else 0)
        total_width = 2 * uniform_grid_size + gap + 2 * padding
        total_height = num_rows * (uniform_grid_size + 30) + (num_rows - 1) * row_gap + 2 * padding
        
        # Create image
        img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw training examples
        for i, example in enumerate(train_examples):
            y_offset = padding + i * (uniform_grid_size + 30 + row_gap)
            
            # Draw training input
            self._draw_grid_centered(draw, np.array(example['input']),
                                   padding, y_offset + 30,
                                   uniform_grid_size, unified_cell_size)
            draw.text((padding, y_offset), f"Training Input", fill='black')
            
            # Draw training output
            output_x = padding + uniform_grid_size + gap
            self._draw_grid_centered(draw, np.array(example['output']),
                                   output_x, y_offset + 30,
                                   uniform_grid_size, unified_cell_size)
            draw.text((output_x, y_offset), f"Training Output", fill='black')
        
        # Draw test input if included
        if test_input is not None:
            test_y_offset = padding + len(train_examples) * (uniform_grid_size + 30 + row_gap)
            self._draw_grid_centered(draw, np.array(test_input),
                                   padding, test_y_offset + 30,
                                   uniform_grid_size, unified_cell_size)
            draw.text((padding, test_y_offset), "Test Input", fill='blue', font=label_font)
        
        img.save(output_path)
        return output_path
    
    def render_problem_with_output(self, 
                                  problem: ARCProblem,
                                  generated_outputs: List[np.ndarray],
                                  output_path: str) -> RenderResult:
        """
        Render problem with both expected and generated outputs
        
        Args:
            problem: ARC problem
            generated_outputs: Generated output grids for each training pair
            output_path: Path to save the image
            
        Returns:
            RenderResult with image and metadata
        """
        train_examples = problem.train_pairs
        test_input = problem.test_pairs[0].x if problem.test_pairs else None
        
        padding = 40
        gap = 100
        row_gap = 100
        uniform_grid_size = 250
        
        # Collect all grids for sizing
        all_grids = []
        for i, pair in enumerate(train_examples):
            all_grids.extend([pair.x, pair.y])  # Input and expected output
            if i < len(generated_outputs) and generated_outputs[i] is not None:
                all_grids.append(generated_outputs[i])  # Generated output
        if test_input is not None:
            all_grids.append(test_input)
        
        # Find optimal cell size
        max_grid_h = max(len(grid) for grid in all_grids if isinstance(grid, np.ndarray))
        max_grid_w = max(len(grid[0]) for grid in all_grids if isinstance(grid, np.ndarray) and len(grid) > 0)
        
        # Calculate cell size with better minimum
        unified_cell_size = min(uniform_grid_size // max_grid_h, 
                               uniform_grid_size // max_grid_w)
        # Ensure minimum readable cell size
        unified_cell_size = max(unified_cell_size, 12)
        # Cap maximum cell size to prevent huge grids for small problems
        unified_cell_size = min(unified_cell_size, 25)
        
        # Layout: 3 grids per row (input, expected output, generated output)
        num_rows = len(train_examples) + (1 if test_input is not None else 0)
        total_width = 3 * uniform_grid_size + 2 * gap + 2 * padding
        total_height = num_rows * (uniform_grid_size + 30) + (num_rows - 1) * row_gap + 2 * padding
        
        # Create image
        img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Load font for labels
        try:
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            label_font = ImageFont.load_default()
        
        # Draw training examples
        for i, pair in enumerate(train_examples):
            y_offset = padding + i * (uniform_grid_size + 30 + row_gap)
            
            # Draw input
            self._draw_grid_centered(draw, pair.x, 
                                   padding, y_offset + 30,
                                   uniform_grid_size, unified_cell_size)
            draw.text((padding, y_offset), f"Input {i+1}", fill='black', font=label_font)
            
            # Draw expected output
            x_expected = padding + uniform_grid_size + gap
            self._draw_grid_centered(draw, pair.y,
                                   x_expected, y_offset + 30,
                                   uniform_grid_size, unified_cell_size)
            draw.text((x_expected, y_offset), f"Expected Output {i+1}", fill='black', font=label_font)
            
            # Draw generated output
            x_generated = padding + 2 * (uniform_grid_size + gap)
            if i < len(generated_outputs) and isinstance(generated_outputs[i], np.ndarray):
                self._draw_grid_centered(draw, generated_outputs[i],
                                       x_generated, y_offset + 30,
                                       uniform_grid_size, unified_cell_size)
                draw.text((x_generated, y_offset), f"Generated Output {i+1}", fill='green', font=label_font)
            else:
                # Draw error message
                draw.text((x_generated + 10, y_offset + uniform_grid_size // 2), 
                         "ERROR", fill='red', font=label_font)
                draw.text((x_generated, y_offset), f"Generated Output {i+1}", fill='red', font=label_font)
        
        # Draw test input if present
        if test_input is not None:
            test_y_offset = padding + len(train_examples) * (uniform_grid_size + 30 + row_gap)
            self._draw_grid_centered(draw, test_input,
                                   padding, test_y_offset + 30,
                                   uniform_grid_size, unified_cell_size)
            draw.text((padding, test_y_offset), "Test Input", fill='blue', font=label_font)
        
        # Save image
        img.save(output_path)
        
        # Prepare metadata
        grid_info = {
            'num_training_pairs': len(train_examples),
            'has_test_input': test_input is not None,
            'cell_size': unified_cell_size,
            'grid_size': uniform_grid_size,
            'generated_outputs': len([g for g in generated_outputs if isinstance(g, np.ndarray)])
        }
        
        return RenderResult(
            image_path=output_path,
            image=img,
            grid_info=grid_info
        )
    
    def _draw_grid_centered(self, draw: ImageDraw.Draw, grid: np.ndarray,
                          x: int, y: int, 
                          box_size: int, cell_size: int):
        """Draw a grid centered within a box"""
        if not isinstance(grid, np.ndarray):
            return
            
        grid_h, grid_w = grid.shape
        display_width = grid_w * cell_size
        display_height = grid_h * cell_size
        
        # Center the grid within the box
        offset_x = (box_size - display_width) // 2
        offset_y = (box_size - display_height) // 2
        
        # Draw the grid
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                x1 = x + offset_x + j * cell_size
                y1 = y + offset_y + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                # Get color
                color = self.colors[int(val) % len(self.colors)]
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='gray')
    
    def render_simple_grid(self, grid: np.ndarray, output_path: str) -> Image.Image:
        """Render a single grid as an image"""
        cell_size = 20
        padding = 10
        
        h, w = grid.shape
        img_width = w * cell_size + 2 * padding
        img_height = h * cell_size + 2 * padding
        
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        for i, row in enumerate(grid):
            for j, val in enumerate(row):
                x1 = padding + j * cell_size
                y1 = padding + i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                
                color = self.colors[int(val) % len(self.colors)]
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='gray')
        
        img.save(output_path)
        return img
    
    def render_training_comparison(self, 
                                  problem: ARCProblem,
                                  execution_result,
                                  output_path: str) -> str:
        """
        Render training examples with expected vs actual outputs for GLM evaluation
        """
        train_pairs = problem.train_pairs
        output_grids = execution_result.output_grids
        
        padding = 40
        gap = 100
        row_gap = 80
        uniform_grid_size = 250  # Increased grid size for better visibility
        
        # Calculate optimal cell size
        all_grids = []
        for pair in train_pairs:
            all_grids.extend([pair.x, pair.y])
        all_grids.extend([grid for grid in output_grids if isinstance(grid, np.ndarray)])
        
        max_grid_h = max(len(grid) for grid in all_grids if isinstance(grid, np.ndarray))
        max_grid_w = max(len(grid[0]) for grid in all_grids if isinstance(grid, np.ndarray) and len(grid) > 0)
        
        # Calculate cell size with better minimum
        unified_cell_size = min(uniform_grid_size // max_grid_h, 
                               uniform_grid_size // max_grid_w)
        # Ensure minimum readable cell size
        unified_cell_size = max(unified_cell_size, 12)
        # Cap maximum cell size to prevent huge grids for small problems
        unified_cell_size = min(unified_cell_size, 25)
        
        # Calculate text size based on grid size
        font_size = max(36, uniform_grid_size // 6)  # Increased font size
        header_font_size = max(48, uniform_grid_size // 4)  # Larger header font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", header_font_size)
        except:
            try:
                # Try alternative font paths
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", header_font_size)
            except:
                font = ImageFont.load_default()
                header_font = ImageFont.load_default()
        
        # Layout: 3 columns (input, expected, actual) x num_training_pairs rows
        num_rows = len(train_pairs)
        total_width = 3 * uniform_grid_size + 2 * gap + 2 * padding
        header_height = 80  # Space for header text
        total_height = header_height + num_rows * (uniform_grid_size + 80) + (num_rows - 1) * row_gap + 2 * padding
        
        # Create image
        img = Image.new('RGB', (total_width, total_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw header
        header_y = 20
        draw.text((padding + uniform_grid_size // 4, header_y), "INPUT", font=header_font, fill='black')
        draw.text((padding + uniform_grid_size + gap + uniform_grid_size // 6, header_y), "EXPECTED", font=header_font, fill='blue')
        draw.text((padding + 2 * (uniform_grid_size + gap) + uniform_grid_size // 6, header_y), "ACTUAL", font=header_font, fill='red')
        
        # Draw training examples
        for i, pair in enumerate(train_pairs):
            y_offset = header_height + padding + i * (uniform_grid_size + 80 + row_gap)
            
            # Draw input
            self._draw_grid_centered(draw, pair.x, 
                                   padding, y_offset,
                                   uniform_grid_size, unified_cell_size)
            
            # Draw expected output
            x_expected = padding + uniform_grid_size + gap
            self._draw_grid_centered(draw, pair.y,
                                   x_expected, y_offset,
                                   uniform_grid_size, unified_cell_size)
            
            # Draw actual output
            x_actual = padding + 2 * (uniform_grid_size + gap)
            if i < len(output_grids) and isinstance(output_grids[i], np.ndarray):
                self._draw_grid_centered(draw, output_grids[i],
                                       x_actual, y_offset,
                                       uniform_grid_size, unified_cell_size)
                
                # Check if they match
                if np.array_equal(pair.y, output_grids[i]):
                    status_text = "✅ MATCH"
                    status_color = 'green'
                else:
                    status_text = "❌ MISMATCH"
                    status_color = 'red'
                    
                draw.text((x_actual, y_offset + uniform_grid_size + 10), 
                         status_text, font=font, fill=status_color)
            else:
                # Draw error message
                error_text = "ERROR"
                draw.text((x_actual + uniform_grid_size//4, y_offset + uniform_grid_size//2), 
                         error_text, font=font, fill='red')
        
        img.save(output_path)
        return output_path