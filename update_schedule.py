# update_schedule.py

import re
import datetime
import nbformat
from nbformat import v4
try:
    import ipynbname
except ImportError:
    raise ImportError("Please install ipynbname (pip install ipynbname) to use update_schedule()")


#####################
# A simple Node class for schedule lines.
#####################
class Node:
    def __init__(self, indent, bullet, checkbox, text):
        self.indent = indent  # number of leading spaces (we ignore the exact spacing later)
        self.bullet = bullet  # e.g., "1." or "*" (if any)
        self.checkbox = checkbox  # either "x" (complete), "" (incomplete) or None (a note line)
        self.text = text.strip()
        self.children = []

    def copy_with(self, checkbox=None, children=None):
        """Return a shallow copy with possibly replaced checkbox and children."""
        new = Node(self.indent, self.bullet, self.checkbox if checkbox is None else checkbox, self.text)
        new.children = children if children is not None else []
        return new


#####################
# Parsing and unparsing
#####################
def parse_schedule(lines):
    """
    Given a list of lines (strings) of the schedule (after the header)
    return a list of Node objects representing the list–tree.
    """
    nodes = []
    stack = []  # will store (node, indent) pairs
    for line in lines:
        if not line.strip():
            continue  # skip blank lines
        # The expected line format is:
        #   <indent><bullet> <optional_checkbox> <text>
        # For example:
        #   "1. [x] Prepare the day and review"
        #   "    * [ ] Ready for training"
        #   "        * Some note text"
        pattern = r"^(\s*)([-*+]|[0-9]+\.)\s+(?:(\[[ xX]\])\s+)?(.*)$"
        m = re.match(pattern, line)
        if m:
            indent_str, bullet, checkbox_part, text = m.groups()
            indent = len(indent_str)
            if checkbox_part is not None:
                # Normalize checkbox: if letter x or X appears, mark as complete.
                if "x" in checkbox_part.lower():
                    checkbox = "x"
                else:
                    checkbox = ""  # empty checkbox meaning not done
            else:
                checkbox = None  # note line
            node = Node(indent, bullet, checkbox, text)
        else:
            # if the line does not match our expected pattern, treat it as a note with its current indent.
            indent = len(line) - len(line.lstrip())
            node = Node(indent, "", None, line.strip())
        # Determine where to put the node in the tree.
        # (Assume that a node whose indent is greater than the previous one is a child.)
        while stack and stack[-1][1] >= node.indent:
            stack.pop()
        if stack:
            stack[-1][0].children.append(node)
        else:
            nodes.append(node)
        stack.append((node, node.indent))
    return nodes


def unparse_schedule(nodes, level=0):
    """
    Given a list of Node objects, return a list of markdown lines.
    We use a fixed indent (4 spaces per level) for output.
    """
    lines = []
    indent = "    " * level
    for node in nodes:
        # If this node is a task (has a checkbox) then output the checkbox.
        if node.checkbox is not None:
            line = f"{indent}{node.bullet} [{'x' if node.checkbox=='x' else ' '}] {node.text}"
        else:
            # note line (or a list item without a checkbox)
            if node.bullet:
                line = f"{indent}{node.bullet} {node.text}"
            else:
                line = f"{indent}{node.text}"
        lines.append(line)
        if node.children:
            lines.extend(unparse_schedule(node.children, level+1))
    return lines


#####################
# Filters for yesterday’s (completed) and next day’s (incomplete) schedules.
#####################
def filter_completed(nodes):
    """
    For yesterday’s schedule: Keep only those nodes (and their children) that were checked.
    (For tasks with checkboxes: if complete, include it; if incomplete, drop it along with all its children.
     However, any note lines (without checkboxes) are always kept.)
    """
    new_nodes = []
    for node in nodes:
        if node.checkbox is None:
            # a note line: keep it.
            new_node = node.copy_with()
            new_node.children = filter_completed(node.children)
            new_nodes.append(new_node)
        else:
            if node.checkbox == "x":
                # This task was completed. Keep it.
                new_children = []
                for child in node.children:
                    if child.checkbox is None:
                        # note lines always come along.
                        new_child = child.copy_with()
                        new_child.children = filter_completed(child.children)
                        new_children.append(new_child)
                    else:
                        if child.checkbox == "x":
                            new_child = child.copy_with()
                            new_child.children = filter_completed(child.children)
                            new_children.append(new_child)
                        # if incomplete, drop child (and its children)
                new_nodes.append(node.copy_with(children=new_children))
            # else: if task was incomplete, do not include it.
    return new_nodes


def filter_for_next_day(nodes, is_top_level=True):
    """
    For the next day’s schedule we want to “reset” tasks.
      - All top–level tasks are always carried over, with their checkboxes emptied.
      - For non–top–level tasks, only those tasks that were incomplete are carried over.
      - (Exception: if a non–top–level task was done but has children that survive, then we carry it as a container
         but without any note–children.)
    """
    new_nodes = []
    for node in nodes:
        if node.checkbox is None:
            # note line: include as is.
            new_node = node.copy_with()
            new_node.children = filter_for_next_day(node.children, is_top_level=False)
            new_nodes.append(new_node)
        else:
            if is_top_level:
                # Always include top–level tasks (reset the checkbox to empty).
                new_node = node.copy_with(checkbox="")
                new_node.children = filter_for_next_day(node.children, is_top_level=False)
                new_nodes.append(new_node)
            else:
                # For non–top–level nodes:
                if node.checkbox == "":
                    # Incomplete task: keep it.
                    new_node = node.copy_with(checkbox="")
                    new_node.children = filter_for_next_day(node.children, is_top_level=False)
                    new_nodes.append(new_node)
                elif node.checkbox == "x":
                    # Completed non–top–level tasks are dropped unless they have at least one child to carry over.
                    filtered_children = filter_for_next_day(node.children, is_top_level=False)
                    # In this case we drop any note lines from a “completed” task.
                    filtered_children = [child for child in filtered_children if child.checkbox is not None]
                    if filtered_children:
                        new_node = node.copy_with(checkbox="")  # reset to incomplete
                        new_node.children = filtered_children
                        new_nodes.append(new_node)
                    # else: drop the node completely.
    return new_nodes


#####################
# The main update_schedule() function.
#####################
def update_schedule():
    """
    This function reads the current notebook, and assuming that the last three cells are:
      1. A markdown cell with a header "## Day X, DD.MM.YYYY"
      2. A markdown cell containing the current day’s schedule.
      3. A code cell that calls update_schedule()
    it will replace these last three cells with five cells:
      - The unchanged previous day header.
      - A new markdown cell showing the previous day’s schedule but with only completed tasks (and note–children).
      - A new markdown cell with header "## Day X+1, DD.MM.YYYY" (with date advanced by one day).
      - A new markdown cell with the next day’s schedule (only the tasks that weren’t done, with checkboxes emptied).
      - The unchanged update_schedule() cell.
    """
    # Locate the current notebook.
    nb_path = ipynbname.path()
    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)
    
    # Check that there are at least three cells.
    if len(nb.cells) < 3:
        raise Exception("Notebook does not have enough cells for schedule update.")
    
    # Assume the last three cells are: previous day header, schedule, and update cell.
    prev_header_cell = nb.cells[-3]
    prev_schedule_cell = nb.cells[-2]
    update_cell = nb.cells[-1]
    
    # Extract the previous day number and date from the header.
    # Expected format: "## Day X, DD.MM.YYYY"
    header_pattern = r"## Day (\d+), (\d{2}\.\d{2}\.\d{4})"
    m = re.match(header_pattern, prev_header_cell.source.strip())
    if not m:
        raise Exception("Header cell format not recognized. Expected '## Day X, DD.MM.YYYY'.")
    prev_day = int(m.group(1))
    prev_date = datetime.datetime.strptime(m.group(2), "%d.%m.%Y")
    
    # Compute next day’s header.
    new_day = prev_day + 1
    new_date = prev_date + datetime.timedelta(days=1)
    new_header_text = f"## Day {new_day}, {new_date.strftime('%d.%m.%Y')}"
    
    # The schedule cell is assumed to have a header line (e.g., "### Schedule:")
    # and then the list of tasks.
    schedule_lines = prev_schedule_cell.source.splitlines()
    header_lines = []
    body_lines = []
    header_done = False
    for line in schedule_lines:
        if not header_done and line.strip().startswith("###"):
            header_lines.append(line)
        else:
            header_done = True
            body_lines.append(line)
    # (If there is an empty line after the header, that’s fine.)
    
    # Parse the body of the schedule.
    tree = parse_schedule(body_lines)
    
    # Create the two versions.
    tree_prev = filter_completed(tree)
    tree_next = filter_for_next_day(tree)
    
    # Reconstruct the markdown text.
    new_prev_schedule_text = "\n".join(header_lines + [""] + unparse_schedule(tree_prev))
    new_next_schedule_text = "\n".join(header_lines + [""] + unparse_schedule(tree_next))
    
    # Build new cells.
    new_cells = nb.cells[:-3]  # all cells before the last three
    
    # (1) Previous day header (unchanged)
    new_cells.append(prev_header_cell)
    # (2) New previous–day schedule (filtered)
    new_cells.append(v4.new_markdown_cell(source=new_prev_schedule_text))
    # (3) New header for current day.
    new_cells.append(v4.new_markdown_cell(source=new_header_text))
    # (4) Next day schedule.
    new_cells.append(v4.new_markdown_cell(source=new_next_schedule_text))
    # (5) The update_schedule() code cell (unchanged)
    new_cells.append(update_cell)
    
    nb.cells = new_cells
    nbformat.write(nb, nb_path)
    
    print(f"Schedule updated. New day: {new_header_text}")
