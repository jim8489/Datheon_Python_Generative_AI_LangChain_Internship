
from typing import Optional
from pydantic import BaseModel, Field, EmailStr


class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    graduation_year: Optional[str] = None


class Experience(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    duration: Optional[str] = None
    description: Optional[str] = None


class ResumeData(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None

    location: Optional[str] = None
    professional_summary: Optional[str] = None

    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)

    certifications: list[str] = Field(default_factory=list)
