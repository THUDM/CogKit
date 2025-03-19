# -*- coding: utf-8 -*-


from pydantic import BaseModel, ConfigDict


class ResponseBody(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True, validate_default=True)
