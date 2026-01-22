You will be given as input a programming *problem*, *code* implementing a solution, and the description of a *misconception*. A misconception is a false belief that the student holds about some programming language construct.

Modify the code such that it looks as if written by a student who has that misconception. **CRITICAL: The student genuinely believes that this modified code solves the problem correctly.** Given their misconception, they are confident that their approach is not only valid but the right way to solve the problem. The student does not suspect any issues with their code - from their perspective, it should work perfectly. You may also rewrite the solution or heavily modify it such that it exhibits the misconception. 


## Important Note

**Misconceptions do not always result in buggy code!** Some misconceptions lead to stylistic differences or inefficiencies rather than errors. For example, a student who believes "all variable identifiers must use only one letter" might write working but less readable code.


## Input Format

**Problem Description:**
[problem_description]

**Problem Title:** [problem_title]

**Given Implementation:**
```py
[correct_solution]
```

**Misconception Description:**
[misconception_desc]


## Output Format

Structure your response using these XML tags:

**Metadata Requirements:**
- **misconception_type**: Choose `benign` or `harmful`
  - **Benign**: Misconceptions that do not cause incorrect program behavior or bugs (style-related issues, suboptimal but functional code patterns, coding practices that work but are not best practices)
  - **Harmful**: Misconceptions that cause incorrect program behavior, logical errors, compilation errors, or runtime errors (logical errors in understanding language constructs, incorrect usage of functions or methods, misunderstanding of data types or operations)  
- **error_type**: Choose `logical`, `syntax`, `runtime`, or `NONE`

```xml
<reasoning>
[Your step-by-step thought process as the student who holds the misconception]
</reasoning>

<code>
[The complete modified Python code exhibiting the misconception]
</code>

<metadata>
misconception_type: [benign|harmful]
error_type: [logical|syntax|runtime|NONE]
modification_notes:[Any additional information about the modification]
</metadata>
```

If the misconception relates to language constructs, concepts, or operations that are not present or relevant in the given problem/solution (e.g., a misconception about loops when the solution doesn't use loops), output:

```xml
<reasoning>
[Explain why the misconception doesn't apply]
</reasoning>

<code>
NONE
</code>

<metadata>
inapplicable: true
reason: [brief explanation]
misconception_type: N/A
error_type: N/A
modification_notes: N/A
</metadata>
```


## Remember

- **Think step-by-step as a student with the misconception who is confident their code is correct**
- **The student must believe their modified code will work and solve the problem properly**
- Make minimal, targeted changes that demonstrate the misconception
- Preserve the overall solution approach when possible
- The resulting code should look like genuine student work
- Not all misconceptions produce errors - some produce working but suboptimal code
- **From the student's perspective (with their misconception), the code should seem logically sound**
- Do NOT add in-line comments in the code. ALL your explanations and beliefs regarding the changes must be in the metadata.

---

## Now Let's Begin with the Actual Task

Using the guidelines and examples above, please analyze the following problem and generate code that exhibits the specified misconception: 


**Problem Description:**
{problem_description}

**Problem Title:** {problem_title}

**Given Implementation:**
```py
{correct_solution}
```

**Misconception Description:**
{misconception_desc}
