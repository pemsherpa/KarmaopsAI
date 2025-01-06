from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

# Create the Base class and automatically generate metadata
Base = declarative_base()


# Example model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)


# Define the MetaData object (this is provided by the Base class)
metadata = Base.metadata
