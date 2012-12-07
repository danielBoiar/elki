package de.lmu.ifi.dbs.elki.database.ids.integer;

/*
 This file is part of ELKI:
 Environment for Developing KDD-Applications Supported by Index-Structures

 Copyright (C) 2012
 Ludwig-Maximilians-Universität München
 Lehr- und Forschungseinheit für Datenbanksysteme
 ELKI Development Team

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import gnu.trove.list.TIntList;
import de.lmu.ifi.dbs.elki.database.ids.ArrayDBIDs;
import de.lmu.ifi.dbs.elki.database.ids.DBID;
import de.lmu.ifi.dbs.elki.database.ids.DBIDIter;
import de.lmu.ifi.dbs.elki.database.ids.DBIDRef;
import de.lmu.ifi.dbs.elki.database.ids.DBIDUtil;
import de.lmu.ifi.dbs.elki.database.ids.DBIDVar;
import de.lmu.ifi.dbs.elki.logging.LoggingUtil;

/**
 * Abstract base class for GNU Trove array based lists.
 * 
 * @author Erich Schubert
 * 
 * @apiviz.has IntegerDBID
 * @apiviz.has DBIDItr
 */
public abstract class TroveArrayDBIDs implements ArrayDBIDs, IntegerDBIDs {
  /**
   * Get the array store.
   * 
   * @return the store
   */
  protected abstract TIntList getStore();

  @Override
  public IntegerDBIDArrayMIter iter() {
    return new DBIDItr(getStore());
  }

  @Override
  public DBID get(int index) {
    return new IntegerDBID(getStore().get(index));
  }

  @Override
  public void assign(int index, DBIDVar var) {
    if (var instanceof IntegerDBIDVar) {
      ((IntegerDBIDVar)var).internalSetIndex(getStore().get(index));
    } else {
      // Much less efficient:
      var.set(get(index));
    }
  }

  @Override
  public int size() {
    return getStore().size();
  }

  @Override
  public boolean isEmpty() {
    return getStore().isEmpty();
  }

  @Override
  public boolean contains(DBIDRef o) {
    return getStore().contains(DBIDUtil.asInteger(o));
  }

  @Override
  public int binarySearch(DBIDRef key) {
    return getStore().binarySearch(DBIDUtil.asInteger(key));
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append('[');
    for(DBIDIter iter = iter(); iter.valid(); iter.advance()) {
      if(buf.length() > 1) {
        buf.append(", ");
      }
      buf.append(((IntegerDBIDRef) iter).internalGetIndex());
    }
    buf.append(']');
    return buf.toString();
  }

  /**
   * Iterate over a Trove IntList, ELKI/C-style.
   * 
   * @author Erich Schubert
   * 
   * @apiviz.exclude
   */
  protected static class DBIDItr implements IntegerDBIDArrayMIter {
    /**
     * Current position.
     */
    int pos = 0;

    /**
     * The actual store we use.
     */
    TIntList store;

    /**
     * Constructor.
     * 
     * @param store The actual trove store
     */
    public DBIDItr(TIntList store) {
      super();
      this.store = store;
    }

    @Override
    public boolean valid() {
      return pos < store.size() && pos >= 0;
    }

    @Override
    public void advance() {
      pos++;
    }

    @Override
    public void advance(int count) {
      pos += count;
    }

    @Override
    public void retract() {
      pos--;
    }

    @Override
    public void seek(int off) {
      pos = off;
    }

    @Override
    public int getOffset() {
      return pos;
    }

    @Override
    public int internalGetIndex() {
      return store.get(pos);
    }

    @Override
    public void remove() {
      store.removeAt(pos);
      pos--;
    }
    
    @Override
    public int hashCode() {
      // Since we add a warning to 'equals', we also override hashCode.
      return super.hashCode();
    }

    @Override
    public boolean equals(Object other) {
      if(other instanceof DBID) {
        LoggingUtil.warning("Programming error detected: DBIDItr.equals(DBID). Use DBIDUtil.equal(iter, id)!", new Throwable());
      }
      return super.equals(other);
    }

    @Override
    public String toString() {
      return Integer.toString(internalGetIndex());
    }
  }
}