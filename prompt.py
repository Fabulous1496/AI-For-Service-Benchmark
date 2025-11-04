PROMPT_EXAMPLE = """
**CORE INSTRUCTIONS:**
1. **Monitor Continuously**: Track the video stream for meaningful event transitions
2. **Output Only on Change**: Generate output ONLY when a new main event type begins
3. **Strict Event Typing**: Event types must be from: [social, housekeeping, cooking, shopping, dining, party, arts & craftwork, leisure, games, music & dance, outing, setup, meeting, commuting]

**OUTPUT REQUIREMENTS:**
For each detected event change, output EXACTLY in this format:
event: [event_type]
video_time_str: [HH:MM:SS]
services: [Service_Category] [Specific service description that aligns with AI agent capabilities]

**SERVICE CATEGORIES & EXAMPLES:**
- **[Find]**: Information retrieval, explanation, summarization
  *Example: "Find recipe for chocolate cake with available ingredients"*
- **[Do]**: Task execution, command implementation  
  *Example: "Set timer for 25 minutes for baking"*
- **[Plan]**: Complex decision support, itinerary planning
  *Example: "Plan weekly meal prep schedule considering dietary restrictions"*

**CRITICAL CONSTRAINTS:**
- Time format MUST be "HH:MM:SS" (e.g., "11:10:00")
- Services must be realistic for current AI agents
- Each event change gets exactly ONE most relevant service suggestion
- Maintain temporal continuity - don't output the same event type consecutively

**EXAMPLE OUTPUTS:**
event: cooking
video_time_str: 11:10:00
services: [Find] Look up step-by-step instructions for proper knife techniques

event: shopping
video_time_str: 11:45:30
services: [Plan] Create optimized shopping route through the store based on shopping list

**REMINDER:** Output immediately when event changes, keep responses concise and machine-parsable for benchmark evaluation.
"""