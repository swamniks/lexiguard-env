#!/usr/bin/env python
"""Test script to verify OpenEnv can discover tasks and graders"""

def test_imports():
    print("Testing imports...")
    try:
        from env.grader import grade_clause_identification, grade_risk_classification, grade_contract_negotiation
        print("✓ All grader functions importable")
        
        from env.tasks import TASKS
        print(f"✓ Found {len(TASKS)} tasks: {[t.task_id for t in TASKS]}")
        
        assert len(TASKS) >= 3, "Need at least 3 tasks"
        print("✓ Task count passes")
        
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def test_grader_scores():
    print("\nTesting grader scores...")
    from env.models import Action
    from env.grader import grade_clause_identification, grade_risk_classification, grade_contract_negotiation
    
    # Test clause_identification
    action = Action(task_id="clause_identification", response="This is a termination clause")
    score = grade_clause_identification(action)
    print(f"  clause_identification score: {score}")
    assert 0.0 <= score <= 1.0, "Score must be between 0 and 1"
    
    # Test risk_classification
    action = Action(task_id="risk_classification", response="High risk due to unlimited liability")
    score = grade_risk_classification(action)
    print(f"  risk_classification score: {score}")
    assert 0.0 <= score <= 1.0, "Score must be between 0 and 1"
    
    # Test contract_negotiation
    action = Action(task_id="contract_negotiation", response="Add liability cap of 12 months fees")
    score = grade_contract_negotiation(action)
    print(f"  contract_negotiation score: {score}")
    assert 0.0 <= score <= 1.0, "Score must be between 0 and 1"
    
    print("✓ All grader scores valid")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("OpenEnv Validation Test")
    print("=" * 50)
    
    if test_imports() and test_grader_scores():
        print("\n✓✓✓ All tests passed! Ready for submission.")
    else:
        print("\n✗✗✗ Tests failed. Fix issues before submitting.")