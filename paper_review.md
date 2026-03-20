# Meticulous Review: "False Precision in Art Market Valuation"

## For Richard Saldanha — Pre-submission Review

**Paper:** "False Precision in Art Market Valuation: Expert Consensus vs. Machine Learning Uncertainty Quantification — A Replication Study"
**Authors:** Bloomsbury Tech
**Reviewed:** 24 February 2026

---

## CRITICAL ISSUES (Must Fix Before Sending to Richard)

### 1. The Fundamental Comparison Problem — Expert Ranges Are Not Prediction Intervals

This is the single biggest issue in the paper. The entire analysis compares expert [low, high] ranges to ML [q₀.₁₀, q₀.₉₀] prediction intervals as if they are the same kind of object. They are not.

**The ML intervals** are explicitly calibrated 80% prediction intervals — they are designed to capture 80% of outcomes by construction (pinball loss at α = 0.10 and α = 0.90).

**Sotheby's expert ranges** are pre-sale estimates. They serve multiple functions simultaneously: guiding bidders, setting reserve prices, managing consignor expectations, marketing. They were never designed or claimed to be 80% prediction intervals. Sotheby's never said "we expect 80% of hammer prices to fall within this range."

Therefore, the finding that expert ranges achieve only 29.4% coverage is measuring experts against a criterion they never claimed to optimise for. The paper frames this as "false precision" — implying experts are attempting and failing to provide calibrated uncertainty. But they may not be attempting that at all.

**What Richard will say:** "You're comparing apples to oranges. What coverage probability are expert ranges *intended* to achieve? If Sotheby's treats them as a 'likely range' with no specific probability attached, your entire framing of 'false precision' collapses. The experts aren't imprecise — they're doing something categorically different from uncertainty quantification."

**Fix:** Either (a) find documentation from Sotheby's or the auction industry defining what coverage the estimate range is intended to provide, (b) reframe the paper away from "false precision" toward "information content comparison" — comparing how much uncertainty information each method conveys, or (c) acknowledge explicitly that expert ranges may not be intended as probabilistic intervals and reframe the analysis as: "regardless of intention, here is the actual coverage achieved, and here is what properly calibrated intervals look like for comparison." Option (c) is the most honest and still makes the paper's point — just less dramatically.

### 2. "A Replication Study" — But What Is Being Replicated?

The subtitle says "A Replication Study" and the abstract mentions "the original paper's qualitative pattern." But the paper never clearly cites what original study is being replicated. Section 1.2 says "This study replicates the seminal work on false precision" — which seminal work? There is no reference. This is a glaring omission. A mathematician will immediately ask: replication of what, by whom, on what data?

**Fix:** Either cite the original study explicitly (by author, title, year) in the abstract and introduction, or drop the "Replication Study" framing entirely if this is actually a new study.

### 3. Spread Formulas Are Incompatible — The 25.2× Claim Is Potentially Invalid

This is a serious mathematical consistency issue.

**Expert spread** (Equation 5):
$$S_{\text{expert}} = \frac{\text{high} - \text{low}}{y} \times 100\%$$

This is computed in *original price space*, normalized by realized price y.

**Model spread** (Equation 6):
$$S_{\text{model}} = \left[\exp(\hat{q}_{0.90} - \hat{q}_{0.10}) - 1\right] \times 100\%$$

This is computed from *log-space* quantile predictions, back-transformed via exponentiation.

These two formulas compute fundamentally different quantities. Expert spread is a ratio of price-space width to realized price. Model spread is the relative width of a log-space interval, which is a ratio of the upper bound to the lower bound minus 1 (since exp(log(a) - log(b)) = a/b).

Specifically: if q̂₀.₉₀ and q̂₀.₁₀ are in log-space, then:
$$S_{\text{model}} = \frac{\hat{Q}_{0.90}}{\hat{Q}_{0.10}} - 1$$

where Q̂ = exp(q̂) are the original-space quantile predictions. This is NOT the same as (Q̂₀.₉₀ - Q̂₀.₁₀) / y.

The paper then claims ML ranges are "25.2× wider" (882.8% / 35.0%). But this ratio is between two quantities computed using **different formulas**. This comparison is only valid if the two formulas produce comparable results, which they don't in general. For the expert spread to be comparable, you'd need:

$$\frac{\text{high} - \text{low}}{y} \approx \frac{\text{high}}{\text{low}} - 1$$

This approximation holds only when high/low ≈ 1 (i.e., small spreads). For the expert spread of 35%, this might be roughly OK. But the claim of direct comparability should be stated and justified, not assumed silently.

**Fix:** Either (a) compute both spreads using the same formula (both as (upper - lower)/y), or (b) explicitly state and justify that the two formulas are approximately comparable for the ranges in question, showing mathematically when the approximation holds and what error it introduces.

### 4. "Mean Bias" in Table 3 vs. Median in Equation 7

Equation (7) defines bias as:
$$\text{Bias} = \text{median}\left(\frac{\hat{q}_{0.50} - y}{y}\right)$$

But Table 3's column header says **"Mean Bias"**. Which is it — mean or median? These give different values, especially in skewed distributions (which art prices certainly are). A mathematician will flag this immediately as sloppy.

**Fix:** Either use median consistently (and relabel Table 3 as "Median Bias") or use mean consistently (and change Equation 7). If you have a reason for using median (robustness to outliers), state it.

### 5. Monotonicity Enforcement Introduces Bias (Equations 2-3)

The paper enforces quantile monotonicity via:
$$\tilde{q}_{0.50} = \max\{\hat{q}_{0.10},\ \hat{q}_{0.50}\}$$
$$\tilde{q}_{0.90} = \max\{\tilde{q}_{0.50},\ \hat{q}_{0.90}\}$$

This is a post-hoc fix that systematically biases the adjusted quantiles upward in cases of crossing. When q̂₀.₅₀ < q̂₀.₁₀ (a crossing violation), you replace the median with the 10th percentile estimate — which is *higher* than the original median estimate. This creates a systematic upward bias in the corrected median for those observations.

More rigorous approaches exist: joint quantile regression (simultaneous estimation with non-crossing constraints), or Chernozhukov et al.'s (2010) rearrangement method, which is provably consistent.

**What Richard will say:** "How often do crossings occur? What fraction of observations require this adjustment? If it's rare (< 1%), mention it and move on. If it's frequent, you have a methodological problem that needs a proper solution."

**Fix:** Report the frequency of quantile crossings across folds. If rare, acknowledge the issue briefly and note it doesn't materially affect results. If frequent (>5%), consider implementing non-crossing quantile regression or rearrangement.

---

## IMPORTANT ISSUES (Should Fix)

### 6. "Stratified" Cross-Validation — Stratified by What?

Section 2.4 says "10-fold stratified cross-validation" but never specifies the stratification variable. For regression problems (continuous target), "stratified" has no standard meaning — stratification requires categorical bins.

If you binned log(hammer_price) into deciles and stratified on those, say so. If you just shuffled and split (which "data shuffled with fixed random seed (42)" suggests), that's ordinary K-fold, not stratified. Using the wrong term undermines credibility with a mathematician.

**Fix:** Either specify exactly what stratification was applied (variable and binning method), or call it "10-fold cross-validation" without "stratified."

### 7. No Formal Statistical Test

The paper's central claim — experts exhibit "false precision" — is never subjected to a formal hypothesis test. You have 13,609 observations and can directly test:

- H₀: Expert coverage ≥ 80% vs. H₁: Expert coverage < 80% (binomial test)
- H₀: Expert coverage = Model coverage vs. H₁: They differ (McNemar's test, since coverage is paired per observation)

The 10-fold CV results implicitly give you this (29.4% with std 1.2% is obviously far from 80%), but stating it formally with a p-value and confidence interval costs nothing and adds rigour. A mathematician expects this.

**Fix:** Add a one-paragraph subsection with formal binomial test for expert coverage and McNemar's test comparing expert vs. model coverage. Report p-values and confidence intervals.

### 8. No Baseline Model

The paper compares experts vs. gradient boosting. But there's no simple baseline to contextualise the ML results. Consider:

- **Naive baseline:** Always predict the training set median, with fixed quantile spread equal to the training set 10th/90th percentile range. What coverage does *that* achieve?
- **Linear quantile regression:** How much of the ML improvement comes from non-linear feature interactions vs. simply using wider, data-calibrated ranges?

Without baselines, the reader can't tell whether the ML model is genuinely learning structure or merely applying appropriately wide intervals.

**Fix:** Add at least one naive baseline (unconditional quantile prediction) to Table 2.

### 9. The 882.8% Spread — Is This Practically Useful?

The paper presents the ML model's median spread of 882.8% as a positive finding (appropriately calibrated). But think about what this means in practice: for a painting that sells for £100,000, the ML prediction interval is approximately £100,000 × 8.83 = £883,000 wide. A range of, say, £20,000 to £903,000.

Is this actionable for anyone? An investor, insurer, or collector? The paper should discuss the *practical utility* of such wide intervals, not just their statistical calibration. There's a well-known trade-off between calibration and sharpness — a perfectly calibrated interval that spans the entire range is statistically correct but informationally useless.

**What Richard will say:** "So your model says this painting will sell for somewhere between £20k and £900k. That's mathematically honest but commercially worthless. What's the sharpness? How does interval width vary across price segments, artists, or lot characteristics?"

**Fix:** Add a discussion of sharpness vs. calibration. Report interval widths by price quantile (are expensive paintings more or less predictable?). Discuss practical utility honestly — this makes the paper stronger, not weaker, because it shows you understand the limitations.

### 10. Authorship

The paper lists "Bloomsbury Tech" as author with no individual names. For academic credibility and for Richard to take this seriously as a research document, it needs named authors. Even industry whitepapers typically have named authors.

**Fix:** Add named authors (you, and anyone else who contributed substantially).

---

## MODERATE ISSUES (Should Address)

### 11. Selection Bias Is Acknowledged but Not Quantified

Section 4 mentions "the data do not include unsold lots" as a limitation. But this isn't just a footnote-worthy limitation — it's a *systematic* bias. Unsold lots are precisely the lots where expert estimates were most wrong (over-estimated, leading to reserves above market willingness to pay). Including them would make expert coverage *even worse*.

But it also means hammer prices are left-truncated at roughly the low estimate (since reserves are typically set near or at the low estimate). This truncation affects quantile regression calibration in a specific, quantifiable direction.

**Fix:** Report how many lots in the original Sotheby's catalogues were *unsold* during this period. Even if you can't include them in the analysis, reporting the fraction gives the reader a sense of how much selection bias might matter.

### 12. "Complete Population" Claim Needs Qualification

The paper claims "the dataset represents the complete qualifying population, not a sample." But the qualifying criteria exclude lots without estimates, without dimensions, without medium metadata. How many lots are excluded by these criteria? If 13,609 is out of 25,000 total Sotheby's lots, that's a 45% exclusion rate and the "complete population" claim is misleading.

**Fix:** Report the total number of lots in Sotheby's catalogues for 2020-2025, the number meeting each inclusion criterion, and the final qualifying count. Show the funnel.

### 13. Feature Importance from Single Fold

Appendix C shows feature importance from Fold 1 only. Feature importance can vary substantially across folds, especially with correlated features. Showing single-fold importance could be misleading.

**Fix:** Either show mean ± std importance across all 10 folds, or at minimum note that Fold 1 is shown for illustration and importance is stable across folds (if it is — check this).

### 14. The 85 Features Claim

Section 2.2 says "85 features" but the enumeration in the text doesn't explicitly add up. Richard will count. Let me count for you:

- Continuous: log(surface_area), log(height), log(width), aspect_ratio, sale_year (= 5)
- Medium: top 15 + "Other" = 16 dummies (or 15 if one is dropped as reference)
- Support: top 5 + "Other" = 6 dummies (or 5)
- Artist: top 50 + "Other" = 51 dummies (or 50)
- Location: 4 sale rooms (or 3 dummies)
- Sale year: "2020-2025" treated as binary flags = 6 (or 5)
- Binary: signed, dated, provenance = 3

Wait — is sale_year continuous OR one-hot? The text lists it both under continuous features ("sale year") and as binary flags ("2020-2025"). It can't be both. This is a contradiction.

If one-hot: 15 + 5 + 50 + 3 + 5 + 3 + 5 = 86 (or similar, depending on reference category drops)
If continuous: 5 + 15 + 5 + 50 + 3 + 3 = 81

**Fix:** Explicitly state the total feature count with a breakdown that adds up. Clarify whether sale_year is continuous or one-hot encoded.

---

## MINOR ISSUES (Nice to Fix)

### 15. Missing Definitions for Standard Notation

- Equation (1): The indicator function 1_{u<0} should be defined. Most readers will know it, but mathematical completeness requires "where 1_{u<0} = 1 if u < 0, and 0 otherwise."
- The paper switches between "hammer price" and y without always defining the mapping explicitly.

### 16. Log-Space to Original-Space Back-Transformation

The paper says models are trained on log(hammer_price) and quantile predictions are back-transformed via exp(). The coverage check should be in original space: does exp(q̂₀.₁₀) ≤ y ≤ exp(q̂₀.₉₀)? This is equivalent to checking q̂₀.₁₀ ≤ log(y) ≤ q̂₀.₉₀ in log-space (since exp is monotone). The paper should state which space the coverage is computed in and confirm equivalence.

### 17. Hyperparameter Tuning

Section 2.3 lists hyperparameters (n_estimators=1000, max_depth=6, etc.) but doesn't say how they were selected. Were they tuned via nested cross-validation? Grid search? Arbitrary choice? If arbitrary, this is fine for a whitepaper but should be stated. If tuned on the test folds, that's data leakage.

### 18. "Seminal" Is a Strong Word

Section 1.2 references "the seminal work on false precision" without citation (Issue #2 above). Beyond the missing citation, "seminal" is a strong claim. Unless the original work is genuinely foundational and widely cited, use "previous" or "original" instead.

### 19. Confidence Interval for Coverage

Table 2 reports coverage mean ± std across folds (29.4 ± 1.2 for experts, 72.0 ± 1.3 for model). Standard deviation across 10 folds is not the same as a confidence interval. With only 10 folds, the standard error of the mean is std/√10 ≈ 0.38. Report this or a proper confidence interval.

### 20. Abstract Overclaims

The abstract says ML models provide "properly calibrated prediction intervals." But 72.0% coverage for a nominal 80% interval is *under*-calibrated. It's much better than experts' 29.4%, but "properly calibrated" means coverage ≈ nominal level. 72% ≠ 80%. Either calibrate better or soften the claim to "substantially better calibrated" or "more realistic."

---

## STRUCTURAL / PRESENTATION

### 21. The Paper's Core Contribution Is Unclear

Is the contribution:
(a) Demonstrating that expert ranges are poorly calibrated? (Interesting but perhaps obvious.)
(b) Showing that ML quantile regression provides better-calibrated intervals? (Incremental.)
(c) Quantifying the gap between expert confidence and actual uncertainty in art markets? (Most interesting.)
(d) A replication study validating previous findings on new data? (If so, cite the original.)

The paper tries to be all four. For Richard, it should be sharp about what's new. The most compelling framing for a mathematician: "Here is the first rigorous quantification of the information content gap between expert judgment and statistical prediction in art markets, using proper scoring rules and out-of-sample evaluation."

### 22. The Discussion Section Is Too Defensive

Section 4 spends a lot of words explaining why expert imprecision might be rational (institutional constraints, herding, etc.). This is good — but it softens the findings so much that the reader wonders: "So is false precision a problem or not?" Be clearer about the normative claim. It's possible to say both "experts have institutional reasons for narrow ranges" AND "this creates a systematic information deficit that has real consequences for market participants."

---

## SUMMARY: Priority Order for Revision

1. **Fix the comparison framing** (#1) — this is the paper's Achilles heel
2. **Fix the spread formula inconsistency** (#3) — mathematical error Richard will catch instantly
3. **Fix the Mean/Median bias contradiction** (#4) — sloppy, easy to fix
4. **Cite the original study or drop "Replication"** (#2)
5. **Add named authors** (#10)
6. **Clarify "stratified" CV** (#6)
7. **Add formal hypothesis test** (#7)
8. **Add baseline model** (#8)
9. **Discuss sharpness vs. calibration** (#9)
10. **Clarify feature count** (#14)
11. **Fix abstract overclaim about "properly calibrated"** (#20)
12. **Report quantile crossing frequency** (#5)
13. Everything else

---

## WHAT RICHARD WILL CARE ABOUT MOST

Richard is a proper mathematician who runs a quant consultancy. He will:

1. **Check the maths first.** The spread formula inconsistency (#3) and mean/median confusion (#4) will damage your credibility before he evaluates the ideas. Fix these before sending anything.

2. **Ask "so what?"** 882.8% prediction intervals (#9) are mathematically honest but commercially useless. He'll want to know: what's the *actionable* output? How does a fund manager or insurer use this? The paper needs a "practical implications" angle.

3. **Probe the comparison fairness.** He will immediately see that expert ranges and ML intervals aren't the same kind of object (#1). Have a clear, honest answer ready.

4. **Want precision in language.** Every term must mean exactly what it says. "Stratified" must mean stratified. "Mean bias" must use the mean. "Replication" must cite what's replicated. "Properly calibrated" must mean coverage ≈ nominal.

5. **Respect intellectual honesty.** A paper that says "here are our limitations, here is what we can and can't conclude" earns more respect than one that overclaims. The paper already does some of this (Section 4 limitations) but needs more rigour in the maths and framing.
