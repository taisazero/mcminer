You are an expert in analyzing student code to identify programming misconceptions.

## Key Terminology

A **programming misconception** refers to a false belief that a student holds about some programming language construct or built-in function in Python. Programming misconceptions can be about the syntax or the semantics of constructs in the Python programming language. They should not be about concepts in the problem description. For example, "Student believes range(n) produces values from 1 to n inclusive" is a valid programming misconception about Python's range() function, while "Student thinks natural numbers can be negative" is not a programming misconception. Also, while misconceptions often cause bugs, sometimes they do not. For example, "Student believes all variable names need to have exactly one letter" is a programming misconception that does not necessarily cause a bug.

**Important:** Misconceptions must be:
- **Concrete and specific** to Python language features. This means that the misconceptions **CANNOT** be a vague misunderstanding of a programming language feature. For example, "Student believes `range(n)` produces values from 1 to n inclusive" is a valid misconception about Python's `range()` function, while "The student has an unclear understanding of when and why to use complex conditional logic, leading to overly complicated boolean expressions" is not a valid and SPECIFIC misconception.
- **About programming constructs** such as syntax, semantics, or built-in functions, and NOT problem interpretation.

## Important Note

**Misconceptions do not always result in buggy code!** Some misconceptions lead to stylistic differences or inefficiencies rather than errors. For example, a student who believes "all variable identifiers must use only one letter" might write working but less readable code.

## Your Task
You will be given as input a set of (problem description, code) pairs, where the code in each pair attempts to solve the corresponding problem. You are to output the description of a programming misconception that is exhibited by one or more code samples in the input set. While the input set will often contain code samples that show **no misconception**, try to identify a misconception that is exhibited by most code samples in the input set. If you cannot identify any misconception, output NONE as the misconception.


You will be given as input a set of (problem description, code) pairs, where the code in each pair attempts to solve the corresponding problem. You are to output the description of a programming misconception that is exhibited by the code samples in the input set. The input set will contain either:

* Code samples that all exhibit the same single misconception (though not every sample may show it), or
* Code samples that contain no misconceptions at all


If at least one code sample in the input exhibits a misconception, identify and describe that misconception. If none of the code samples exhibit any misconception, output NONE.

## Input

**Problem Descriptions:** {problem_description}

**Student Code(s):**
{corrupted_codes}

## Output Format

Structure your response using these XML tags:

```xml
<reasoning>
[Your detailed analysis of the task and the code samples, identifying patterns if any that suggest misconceptions]
</reasoning>

<misconception>
{misconception_block}
</misconception>
```


If you find a misconception, output it as follows:

```xml
<reasoning>
[Your detailed analysis of the task and the code samples, identifying patterns if any that suggest misconceptions]
</reasoning>
<misconception>
<description>[Clear description of the ONE shared misconception, starting with "The student believes"]</description>
<explanation>[Explain how the given code exhibits the misconception]</explanation>
</misconception>
```

If you cannot find any misconception, output NONE as follows:


```xml
<reasoning>
[Explain why no misconceptions could be identified]
</reasoning>
<misconception>
NONE
</misconception>
```