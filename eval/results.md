# Evaluation results

- Embedding model: `BAAI/bge-small-en-v1.5`, top-k 8, relevance gate 0.5
- LLM: `gemini-3.5-flash-lite`

## Summary

| Category | Questions | Retrieval recall@k | Status correct | Fact recall | Citation recall |
|---|---|---|---|---|---|
| answerable | 5 | 100% | 100% | 100% | 100% |
| multi_evidence | 4 | 100% | 75% | 88% | 88% |
| unanswerable | 4 | n/a | 100% | n/a | n/a |

## Per question

| ID | Question | Top score | Retrieval recall | Relevance gate | Status | Facts | Cited docs | Quotes verified |
|---|---|---|---|---|---|---|---|---|
| A1 | How heavy is the Aero X2? | 0.752 | 100% | passed | answered ✓ | 100% | aero_x2_spec_sheet.pdf | 1/1 |
| A2 | How many vacation days do full-time employees get? | 0.861 | 100% | passed | answered ✓ | 100% | employee_handbook.md | 1/1 |
| A3 | Can I send my bike back if I've already ridden it 80 km? | 0.747 | 100% | passed | answered ✓ | 100% | warranty_and_returns.txt | 1/1 |
| A4 | At what charge level does battery-saver mode switch on now? | 0.752 | 100% | passed | answered ✓ | 100% | firmware_release_notes.md | 1/1 |
| A5 | Is it safe to clean the bike with a pressure washer? | 0.578 | 100% | passed | answered ✓ | 100% | aero_x2_spec_sheet.pdf | 1/1 |
| M1 | How long does it take to charge the Aero X2 to 80%, and which firmware version added the option to stop charging at 80%? | 0.744 | 100% | passed | answered ✓ | 100% | aero_x2_spec_sheet.pdf, firmware_release_notes.md | 2/2 |
| M2 | If I register my bike within 60 days, how long is the battery covered, and what is the battery's rated capacity? | 0.792 | 100% | passed | partial ✗ | 50% | warranty_and_returns.txt | 1/1 |
| M3 | As an employee leasing an Aero X2 through the bike plan, what would it cost to buy the bike at the end of the lease? | 0.722 | 100% | passed | answered ✓ | 100% | aero_x2_spec_sheet.pdf, employee_handbook.md | 2/2 |
| M4 | Range Assist says I have 60 km left on a frosty morning below 0 °C. How far could the real range be off, and is a fix coming? | 0.684 | 100% | passed | answered ✓ | 100% | firmware_release_notes.md | 1/1 |
| U1 | Who is the CEO of Velora Cycles? | 0.679 | n/a | passed | unanswerable ✓ | n/a | — | 0/0 |
| U2 | Does the Aero X2 come with a built-in GPS tracker for theft protection? | 0.638 | n/a | passed | unanswerable ✓ | n/a | — | 0/0 |
| U3 | What is the capital of Australia? | 0.45 | n/a | blocked | unanswerable ✓ | n/a | — | 0/0 |
| U4 | How much does the Aero X2 cost in the United States in dollars? | 0.695 | n/a | passed | unanswerable ✓ | n/a | — | 0/0 |

## Answers

**A1. How heavy is the Aero X2?**

> The complete Velora Aero X2 bike weighs 21.4 kg including the battery [1].

**A2. How many vacation days do full-time employees get?**

> Full-time employees receive 25 vacation days per year, in addition to Dutch public holidays [1].

**A3. Can I send my bike back if I've already ridden it 80 km?**

> Yes, you can still return your bike within 30 days if you have ridden it more than 50 km (such as 80 km), but a 10% restocking fee applies [1].

**A4. At what charge level does battery-saver mode switch on now?**

> Battery-saver mode now activates automatically at 15% charge.

**A5. Is it safe to clean the bike with a pressure washer?**

> No, it is not safe to clean the bike with a pressure washer; the motor, battery, and display must not be cleaned with a pressure washer.

**M1. How long does it take to charge the Aero X2 to 80%, and which firmware version added the option to stop charging at 80%?**

> Charging the Aero X2 from empty to 80% takes 2.5 hours, and version 3.4 added the option to cap charging at 80%.

**M2. If I register my bike within 60 days, how long is the battery covered, and what is the battery's rated capacity?**

> If you register your bike within 60 days, you receive an extended battery warranty of 3 years [1].

_Missing:_ The sources do not state the battery's rated capacity (they only mention a rated capacity percentage threshold of 70% and an energy capacity of 540 Wh for the integrated battery in the Aero X2 model).

**M3. As an employee leasing an Aero X2 through the bike plan, what would it cost to buy the bike at the end of the lease?**

> At the end of the 36-month lease, you can buy the bike for 15% of its original retail price [1]. With a recommended retail price of EUR 3,290, the purchase price would be EUR 493.50 [1, 3].

**M4. Range Assist says I have 60 km left on a frosty morning below 0 °C. How far could the real range be off, and is a fix coming?**

> Range Assist estimates can be up to 20% too optimistic in temperatures below 0 °C [1]. A fix is planned for version 3.5 [1].

**U1. Who is the CEO of Velora Cycles?**

> The documents don't contain information to answer this question.

_Missing:_ The provided sources do not contain information about who the CEO of Velora Cycles is.

**U2. Does the Aero X2 come with a built-in GPS tracker for theft protection?**

> The documents don't contain information to answer this question.

_Missing:_ The provided sources do not contain any information about whether the Aero X2 comes with a built-in GPS tracker for theft protection.

**U3. What is the capital of Australia?**

> The documents don't contain information to answer this question.

_Missing:_ No passage in the documents is relevant to this question.

**U4. How much does the Aero X2 cost in the United States in dollars?**

> The documents don't contain information to answer this question.

_Missing:_ The provided sources do not contain the price of the Aero X2 in the United States or in dollars.

