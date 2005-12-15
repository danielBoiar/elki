package de.lmu.ifi.dbs.data;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;


/**
 * RealVector is an abstract implementation of FeatureVector. Provided
 * is an attribute separator (space), and the ID-methods
 * as required for a MetricalObject.
 * 
 * @author Arthur Zimek (<a href="mailto:zimek@dbs.ifi.lmu.de">zimek@dbs.ifi.lmu.de</a>)
 */
public abstract class RealVector<T extends Number> implements FeatureVector<T>
{
    
    
    /**
     * The String to separate attribute values in a String that represents the
     * values.
     */
    public final static String ATTRIBUTE_SEPARATOR = " ";


    /**
     * The unique id of this object.
     */
    private Integer id;
    
    /**
     * Returns the unique id of this RealVector object.
     * 
     * @return the unique id of this RealVector object
     * @see MetricalObject#getID()
     */
    public Integer getID()
    {
        return id;
    }

    /**
     * Sets the id of this RealVector object. The id must be unique within one
     * database.
     * 
     * @param id the id of the object
     * @see MetricalObject#setID(Integer)
     */
    public void setID(Integer id)
    {
        this.id = id;
    }

    /**
     * 
     * 
     * @see de.lmu.ifi.dbs.data.FeatureVector#newInstance(null[])
     */
    public FeatureVector<T> newInstance(T[] values) throws SecurityException, NoSuchMethodException, IllegalArgumentException, InstantiationException, IllegalAccessException, InvocationTargetException
    {
        Class[] parameterClasses = {values.getClass()};
        Object[] parameterValues = {values};
        Constructor c = this.getClass().getConstructor(parameterClasses);
        return (FeatureVector<T>) c.newInstance(parameterValues);
    }

    /**
     * @see MetricalObject#equals(Object)
     */
    abstract public boolean equals(Object obj);


    
}
