import json

class LineIterator:
    """
    A helper class for parsing the byte stream input.

    The output of the model will be in the following format:
    ```
    b'{"outputs": [" a"]}\n'
    b'{"outputs": [" a challenging"]}\n'
    b'{"outputs": [" a challenging problem"]}\n'
    ...
    ```
    """

    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = b''

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # Check if there's a complete JSON object in the buffer
            newline_index = self.buffer.find(b'\n\n')
            if newline_index != -1:
                line = self.buffer[:newline_index]
                self.buffer = self.buffer[newline_index + 1:]

                return line

            # If not, read the next chunk from the byte iterator
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.buffer:
                    # If there are remaining bytes in the buffer, yield them
                    line = self.buffer
                    self.buffer = b''
                    return line
                raise

            # Append the chunk to the buffer
            self.buffer += chunk["PayloadPart"]["Bytes"]

def has_same_prefix(word0, word1):
    if word0 is None:
        return False
    # Get the first word as the reference prefix
    if word1.startswith(word0):
        return True
    return False
