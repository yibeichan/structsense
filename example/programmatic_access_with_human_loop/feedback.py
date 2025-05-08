import logging
from structsense.app import kickoff
from structsense.humanloop import ProgrammaticFeedbackHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def modify_entity_names(data):
    """
    Recursively modify any fields named 'entity', 'entities', 'name', or 'names' to "test".
    """
    print("Starting entity name modification")
    if isinstance(data, dict):
        modified = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                modified[key] = modify_entity_names(value)
            elif key.lower() in {'entity', 'entities', 'name', 'names'}:
                print(f"Modifying field '{key}' to 'test'")
                modified[key] = "test"
            else:
                modified[key] = value
        return modified

    elif isinstance(data, list):
        return [modify_entity_names(item) for item in data]

    else:
        return data

def process_feedback(pending_feedback):
    """
    Process the pending feedback and return a dict with 'choice' and optional 'modified_data'.
    """
    if not pending_feedback:
        logger.warning("No pending feedback to process")
        return {"choice": "1"}

    data = pending_feedback['data']
    step_name = pending_feedback.get('step_name')
    agent_name = pending_feedback.get('agent_name')
    
    print(f"\n{'='*50}")
    print(f"Processing feedback for step: {step_name}")
    print(f"Agent: {agent_name}")
    print(f"Original data type: {type(data)}")
    print(f"Original data: {data}")

    if isinstance(data, (dict)):
        modified = modify_entity_names(data)
        print(f"Modified data: {modified}")
        return {"choice": "3", "modified_data": modified}

    print("Data is not a dict/list, approving without modification")
    return {"choice": "3", "modified_data": data}

def process_with_structsense():
    """
    Runs StructSense kickoff and handles feedback loops.
    Returns the final result or raises on error.
    """
    feedback_handler = ProgrammaticFeedbackHandler()
    try:
        print("Starting StructSense flow...")
        result = kickoff(
            agentconfig="config/ner_agent.yaml",
            taskconfig="config/ner_task.yaml",
            embedderconfig="config/embedding.yaml",
            knowledgeconfig="config/search_ontology_knowledge.yaml",
            agent_feedback_config="config/human_in_loop.yaml",
            input_source="/Users/tekrajchhetri/Downloads/data/test_small.pdf",
            enable_human_feedback = True,
            feedback_handler=feedback_handler
        )
        print(f"\n{'$='*50}")
        print(f"Initial kickoff result: {result}")
        print(f"\n{'=$'*50}")

        # Keep processing feedback until we get a final result
        while result == "feedback":
            print("\n" + "="*50)
            print("Feedback required - processing...")
            
            pending = feedback_handler.get_pending_feedback()
            if pending:
                print("Found pending feedback")
                fb = process_feedback(pending)
                
                print(f"Providing feedback with choice: {fb['choice']}")
                if fb.get("modified_data"):
                    print("Modified data will be used")
                
                result = feedback_handler.provide_feedback(
                    choice=fb["choice"],
                    modified_data=fb.get("modified_data")
                )
                feedback_handler.clear_pending_feedback()
                
                print(f"Feedback processed, new result: {result}")
            else:
                logger.warning("No pending feedback found but result was 'feedback'")
                break

        print("\n" + "="*50)
        print("Flow completed")
        return result

    except Exception as e:
        logger.error(f"Error processing with StructSense: {e}")
        raise

if __name__ == "__main__":
    print("Starting StructSense processing...")
    final_result = process_with_structsense()
    print("\n" + "="*50)
    print("Final result:")
    print(final_result)