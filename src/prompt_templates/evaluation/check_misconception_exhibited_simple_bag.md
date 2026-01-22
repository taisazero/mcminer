You are evaluating whether one or more code samples exhibit a specific programming misconception.

## The Misconception
<misconception>
**Description:** {misconception_description}

**Example:** {misconception_example}
</misconception>

## The Code Samples to Analyze
{code_to_analyze}

## Your Task
Determine whether **ANY** of the code samples above exhibit the misconception described. Following the Multiple Instance Learning principle: if at least one code sample exhibits the misconception, answer Y.

### Key Understanding
**A misconception does NOT necessarily induce bugs or errors!** Code can be:
- **Syntactically correct** (no syntax errors)
- **Logically correct** (produces expected output)
- **Yet still exhibit a misconception** (shows the student holds a false belief)

Misconceptions can be:
- **Benign**: Leading to style issues, inefficiencies, or non-idiomatic patterns while still working correctly
- **Harmful**: Causing incorrect behavior, logical errors, or runtime errors

### Analysis Guidelines

1. **Examine each code sample individually**
   - Look for the misconception pattern in each sample
   - One clear instance is sufficient for Y

2. **Understand the misconception deeply**
   - What incorrect belief does the student have?
   - What coding patterns would reveal this belief?
   - Would this belief lead to different code structure/approach?

3. **Focus on the belief, not the outcome**
   - Does any code structure suggest the student holds this false belief?
   - Even if the code works, does it show the misconception pattern?
   - Would someone with correct understanding write it differently?

### Important Notes
- Answer Y if **ANY** code sample shows patterns consistent with the misconception, even if it works correctly
- The misconception is about what the student believes, not whether their code fails
- Consider whether any code structure reveals the student's mental model
- If multiple samples are provided, finding the pattern in just one is sufficient for Y

## Output Format
<answer>
<exhibits_misconception>Y or N</exhibits_misconception>
<confidence>high|medium|low</confidence>
</answer>
