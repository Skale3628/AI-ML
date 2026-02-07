
# Summary of python_Doc_text_splitter.py :>
# This Python script demonstrates the use of LangChain's RecursiveCharacterTextSplitter to split a sample Python code snippet into manageable chunks based on language-specific rules.

# Imports: Utilizes RecursiveCharacterTextSplitter and Language from langchain_text_splitters.
# Sample Text: Defines a string containing a Student class with __init__, get_details, and is_passing methods, along with example usage code.
# Splitter Configuration: Initializes the splitter for Python language with a chunk size of 300 characters and no overlap.
# Splitting and Output: Splits the text into chunks, then prints the total number of chunks and the content of the second chunk.
# The script outputs the split results, showing how the text is divided (e.g., likely separating class definition from usage code).


from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[1])