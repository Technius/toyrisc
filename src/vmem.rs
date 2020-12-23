use std::cell::{RefCell, UnsafeCell};
use std::collections::HashMap;
use std::ops::{Index, IndexMut, Range};

/// Sparse paged memory data structure
pub struct SparseMem {
    page_size: usize,
    // Invariant: elements should never be removed from data
    data: UnsafeCell<Vec<u8>>,
    pages: RefCell<HashMap<usize, usize>>,
}

impl SparseMem {
    pub fn new(page_size: usize) -> Self {
        SparseMem {
            page_size,
            data: UnsafeCell::new(Vec::new()),
            pages: RefCell::new(HashMap::new())
        }
    }

    pub fn get_page(&self, idx: usize) -> &[u8] {
        let mut pages = self.pages.borrow_mut();
        let page_count = pages.len();
        unsafe {
            let data_idx: usize = pages.entry(idx as usize).or_insert_with(|| {
                (*self.data.get()).extend(vec![0; self.page_size as usize]);
                page_count
            }).clone();
            &(*self.data.get())[data_idx..(data_idx + self.page_size as usize)]
        }
    }

    pub fn get_page_mut(&mut self, idx: usize) -> &mut [u8] {
        let mut pages = self.pages.borrow_mut();
        let page_count = pages.len();
        unsafe {
            let data_idx: usize = pages.entry(idx as usize).or_insert_with(|| {
                (*self.data.get()).extend(vec![0; self.page_size as usize]);
                page_count
            }).clone();
            &mut (*self.data.get())[data_idx..(data_idx + self.page_size as usize)]
        }
    }
}

impl Index<usize> for SparseMem {
    type Output = u8;

    fn index(&self, addr: usize) -> &Self::Output {
        &self.get_page(addr / self.page_size)[addr % self.page_size]
    }
}

impl IndexMut<usize> for SparseMem {
    fn index_mut(&mut self, addr: usize) -> &mut Self::Output {
        let psize = self.page_size;
        &mut self.get_page_mut(addr / self.page_size)[(addr % psize) as usize]
    }
}

pub struct PageTable {
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paging() {
        let mut mem = SparseMem::new(4096);
        let page = 439;

        {
            let page: &mut [u8] = mem.get_page_mut(page);
            for i in 0..32 {
                page[i as usize] = i;
            }
        }

        assert_eq!(mem.get_page(page)[31], 31);
    }
}
