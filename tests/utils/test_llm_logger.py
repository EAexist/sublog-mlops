import logging

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)

class TestLLMLogger(CustomLogger):
    __test__ = False

    def log_pre_api_call(self, model, messages, kwargs):
        logger.info(f"🚀 Calling {model} with {len(messages)} messages")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        # Professional logging of the result
        content = getattr(response_obj.choices[0].message, 'content', 'No Content')
        logger.info(f"✅ Success: {kwargs.get('model')} | Response: {content[:100]}...")

    def log_failure_event(self, kwargs, exception, start_time, end_time):
        # This is where your 'NoneType' or 'JSON' errors will be caught automatically
        logger.error(f"❌ LLM Call Failed: {str(exception)}")
        if 'response_obj' in kwargs:
             logger.error(f"Raw Response: {kwargs['response_obj']}")
