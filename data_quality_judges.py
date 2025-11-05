import dspy

class GeneratedDataQuality(dspy.Signature):
    """You are an expert in assessing data quality. You are provided with a generated data example, hard requirements for this generated data instance to follow, and example data instances from which this generated data example is derived from. Judge the quality of the generated data example along naturalness and adherence to requirements."""
    generated_instance = dspy.InputField(desc="The generated data example")
    hard_requirements = dspy.InputField(desc="The requirements that the generated data should follow")
    data_examples = dspy.InputField(desc="Example data")
    data_naturalness = dspy.OutputField(desc="Is the generated data naturalistic? Be very strict. Respond with yes or no")
    data_adherence = dspy.OutputField(desc="Does the new data instance obey the hard requirements? Be very strict. Respond with yes or no")
    data_feedback = dspy.OutputField(desc="Feedback and advice for improving the naturalness and adherence of the generated data instance")
    
class DataRewrite(dspy.Signature):
    """You are provided with a generated data example, hard requirements for this generated data instance to follow, and feedback for improving data quality. Rewrite this generated instance according to the provided feedback."""
    generated_instance = dspy.InputField(desc="The generated data example")
    hard_requirements = dspy.InputField(desc="The requirements that the generated data should follow")
    data_feedback = dspy.InputField(desc="Feedback and advice for improving the naturalness and adherence of the generated data instance")
    data_rewrite = dspy.OutputField(desc="The generated data instance, rewritten according to the feedback")

class DataQualityJudge(dspy.Module):
    def __init__(self, hard_requirement, callbacks=None):
        super().__init__(callbacks)
        self.requirement = hard_requirement
        self.quality_judge = dspy.ChainOfThought(GeneratedDataQuality)
        self.rewriter = dspy.ChainOfThought(DataRewrite)
        
    def forward(self, generated_instance, data_examples):
        judge_output = self.quality_judge(generated_instance=generated_instance,
                                          hard_requirements=self.requirement,
                                          data_examples=data_examples)
        data_naturalness, data_adherence, data_feedack = judge_output.data_naturalness, judge_output.data_adherence, judge_output.data_feedback
        data_naturalness = data_naturalness.lower().startswith("yes")
        data_adherence = data_adherence.lower().startswith("yes")
        print(data_naturalness, data_adherence, data_feedack)
        if not (data_naturalness and data_adherence):
            generated_instance = self.rewriter(generated_instance=generated_instance,
                                               hard_requirements=self.requirement,
                                               data_feedack=data_feedack).data_rewrite
        return generated_instance
        
        