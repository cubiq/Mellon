import logging

from diffusers import ComponentSpec

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")


class Scheduler(NodeBase):
    label = "Scheduler"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    params = {
        "scheduler_in": {"label": "Scheduler", "display": "input", "type": "scheduler"},
        "scheduler": {
            "label": "Scheduler",
            "options": {"EulerDiscreteScheduler": "Euler", "EulerAncestralDiscreteScheduler": "Euler Ancestral"},
            "value": "EulerDiscreteScheduler",
            "onChange": "updateNode",
        },
        "scheduler_out": {"label": "Scheduler", "display": "output", "type": "scheduler"},
    }

    def updateNode(self, values, ref):
        value = values.get("scheduler")
        key = ref.get("key")
        value_from_key = values.get(key)

        print(f"value: {value}, key: {key}, value_from_key: {value_from_key}")
        print("All values", values)

        params = {}

        if value == "EulerDiscreteScheduler":
            params = {
                "sigmas": {
                    "label": "Sigmas",
                    "type": "string",
                    "options": {
                        "default": "Default",
                        "use_karras_sigmas": "Karras",
                        "use_exponential_sigmas": "Exponential",
                        "use_beta_sigmas": "Beta",
                    },
                    "value": "default",
                },
                "timestep_spacing": {
                    "label": "Timestep Spacing",
                    "type": "string",
                    "options": {
                        "leading": "Leading",
                        "trailing": "Trailing",
                        "linspace": "Linspace",
                    },
                    "value": "leading",
                },
            }

        self.send_node_definition(params)

    def execute(self, scheduler_in, scheduler, **kwargs):
        logger.debug(f" Scheduler ({self.node_id}) received parameters:")
        logger.debug(f" - input_scheduler: {scheduler_in}")
        logger.debug(f" - scheduler: {scheduler}")
        logger.debug(f" - kwargs: {kwargs}")

        scheduler_component = components.get_one(scheduler_in["model_id"])
        scheduler_cls = getattr(__import__("diffusers", fromlist=[scheduler]), scheduler)

        scheduler_options = {}
        for key, value in kwargs.items():
            if key == "sigmas":
                if value != "default":
                    scheduler_options[value] = True
            else:
                scheduler_options[key] = value

        logger.debug(f" - scheduler_options: {scheduler_options}")

        schedule_spec = ComponentSpec(
            name="scheduler",
            type_hint=scheduler_cls,
            config=scheduler_component.config,
            default_creation_method="from_config",
        )
        new_scheduler = schedule_spec.create(**scheduler_options)
        comp_id = components.add(name="scheduler", component=new_scheduler, collection=self.node_id)
        logger.debug(f" Scheduler: new_scheduler: {new_scheduler}")

        return {"scheduler_out": components.get_model_info(comp_id)}
