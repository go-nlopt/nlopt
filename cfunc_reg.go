package nlopt

import (
	"sync"
)

var (
	funcMutex sync.Mutex

	funcMap = make(map[uintptr]interface{})
	funcPtr uintptr
)

func makeFuncPtr(f Func) uintptr {
	return _setFuncMapEntry(f)
}

func makeMfuncPtr(f Mfunc) uintptr {
	return _setFuncMapEntry(f)
}

func _setFuncMapEntry(f interface{}) uintptr {
	funcMutex.Lock()
	defer funcMutex.Unlock()
	funcPtr++
	funcMap[funcPtr] = f
	return funcPtr
}

func getFunc(ptr uintptr) Func {
	return _getFuncMapEntry(ptr).(Func)
}

func getMfunc(ptr uintptr) Mfunc {
	return _getFuncMapEntry(ptr).(Mfunc)
}

func _getFuncMapEntry(ptr uintptr) interface{} {
	funcMutex.Lock()
	defer funcMutex.Unlock()
	return funcMap[ptr]
}

func freeFuncPtr(ptr uintptr) {
	funcMutex.Lock()
	defer funcMutex.Unlock()

	delete(funcMap, ptr)
}
